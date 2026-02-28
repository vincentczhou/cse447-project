#!/usr/bin/env python
"""
Gemini API character-level next-character predictor (async version).

Given partial text strings (one per line), predicts the three most likely
next characters using the Gemini API with structured JSON output (Pydantic).
Uses asyncio + a semaphore for bounded concurrency, exponential-backoff
retries for rate limits / transient errors, and a tqdm progress bar.

Reads GEMINI_API_KEY / GOOGLE_API_KEY from environment or a .env file in
the project root.

Usage:
    python scripts/gemini_predictor.py \
        --input      example/input.txt \
        --output     output/gemini_pred.txt \
        --answer     example/answer.txt   # optional – prints accuracy
        --sample     100                  # optional – randomly sample N inputs
        --concurrency 20                  # optional – max parallel requests
        --verbose                         # optional – per-line grading output
"""

import argparse
import asyncio
from dotenv import load_dotenv
import os
from pathlib import Path
import random
import sys
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator
from tqdm.asyncio import tqdm

# ── Configuration ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"
FALLBACK_PRED = " ea"
CONTEXT_LIMIT = 200
DEFAULT_CONCURRENCY = 20
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5  # seconds; doubles each attempt + jitter
REQUEST_TIMEOUT = 30  # seconds per attempt before retrying

SYSTEM_PROMPT = (
    "You are a character-level language model assisting with next-character prediction.\n"
    "You will be given a partial string from a dialogue utterance. "
    "The input may be in English or another language, or a mix of both.\n"
    "Your task is to predict the three most likely next characters "
    "(letters, digits, spaces, punctuation, etc.) in order of probability.\n"
    "Scoring is caseless, so prefer the lowercase form of a letter when unsure of case.\n"
    "Return a JSON object with exactly three keys: 'first', 'second', 'third'.\n"
    "Each value must be exactly one character. "
    "Use a literal space character when space is your guess.\n"
)


class CharPrediction(BaseModel):
    first: str = Field(description="Most likely next character (exactly one character)")
    second: str = Field(
        description="Second most likely next character (exactly one character)"
    )
    third: str = Field(
        description="Third most likely next character (exactly one character)"
    )

    @field_validator("first", "second", "third")
    @classmethod
    def single_char(cls, v: str) -> str:
        return v[:1] or " "

    def to_pred(self) -> str:
        return self.first + self.second + self.third


def build_prompt(text: str) -> str:
    trimmed = text if len(text) <= CONTEXT_LIMIT else "…" + text[-CONTEXT_LIMIT:]
    return f'Predict the next character for this partial text:\n"{trimmed}"'


async def predict_single(
    client: genai.Client,
    text: str,
    semaphore: asyncio.Semaphore,
    config: types.GenerateContentConfig,
) -> tuple[str, float]:
    async with semaphore:
        t0 = time.perf_counter()
        result = FALLBACK_PRED
        for attempt in range(MAX_RETRIES):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=MODEL_NAME,
                        contents=build_prompt(text),
                        config=config,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                result = CharPrediction.model_validate_json(resp.text or "").to_pred()
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"  [retry {attempt + 1}/{MAX_RETRIES}] {e} — waiting {delay:.1f}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(delay)
                else:
                    print(
                        f"  [error] giving up after {MAX_RETRIES} attempts: {e}",
                        file=sys.stderr,
                    )
        elapsed = time.perf_counter() - t0
    return result, elapsed


async def predict_all(
    client: genai.Client, inputs: list[str], concurrency: int
) -> tuple[list[str], float]:
    print(f"  {len(inputs)} inputs (concurrency={concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        # thinking_config=types.ThinkingConfig(thinking_budget=0),
        # thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        response_mime_type="application/json",
        response_schema=CharPrediction,
    )

    wall_start = time.perf_counter()

    async def _run(idx: int, text: str) -> tuple[int, str, float]:
        pred, elapsed = await predict_single(client, text, semaphore, config)
        return idx, pred, elapsed

    tasks = [_run(i, t) for i, t in enumerate(inputs)]
    results = await tqdm.gather(*tasks, desc="Predicting", unit="input")
    wall_time = time.perf_counter() - wall_start
    preds = [r[1] for r in sorted(results)]
    return preds, wall_time


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def grade(preds: list[str], golds: list[str], verbose: bool = False) -> float:
    correct = 0
    for i, (p, g) in enumerate(zip(preds, golds)):
        hit = g.lower() in p[:3].lower()
        correct += hit
        if verbose:
            print(f"  {'✓' if hit else '✗'} [{i}] gold='{g}' pred='{p}'")
    return correct / len(golds) if golds else 0.0


def main():
    parser = argparse.ArgumentParser(description="Gemini character-level predictor")
    parser.add_argument("--input", required=True, help="Path to input.txt")
    parser.add_argument("--output", default="output/gemini_pred.txt")
    parser.add_argument("--answer", default=None, help="Path to answer.txt (optional)")
    parser.add_argument("--api-key", default=None, help="Gemini API key")
    parser.add_argument(
        "--sample", type=int, default=None, help="Randomly sample N inputs"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip prediction; grade an existing --output file against --answer",
    )
    args = parser.parse_args()

    if args.eval_only:
        if not args.answer:
            sys.exit("Error: --eval-only requires --answer")
        preds = load_lines(args.output)
        golds = load_lines(args.answer)
        print(f"Evaluating {args.output} ({len(preds)} predictions)")
        acc = grade(preds, golds, verbose=args.verbose)
        print(
            f"\n  Accuracy (gold in top-3): {acc:.2%} ({int(acc * len(golds))}/{len(golds)})"
        )
        return

    key = args.api_key or API_KEY
    if not key:
        sys.exit("Error: set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api-key")

    client = genai.Client(api_key=key)
    inputs = load_lines(args.input)
    golds = load_lines(args.answer) if args.answer else None

    if args.sample and args.sample < len(inputs):
        random.seed(42)
        indices = sorted(random.sample(range(len(inputs)), args.sample))
        inputs = [inputs[i] for i in indices]
        if golds:
            golds = [golds[i] for i in indices]
        print(f"Sampled {args.sample}/{len(load_lines(args.input))} inputs (seed=42)")

    print(f"Model : {MODEL_NAME}")
    print(f"Inputs: {len(inputs)}\n")

    preds, elapsed = asyncio.run(predict_all(client, inputs, args.concurrency))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in preds)
    print(f"\nPredictions written to {args.output}")

    avg_ms = elapsed / len(inputs) * 1000 if inputs else 0
    print(f"\n{'=' * 50}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Inputs    : {len(inputs)}")
    print(f"  Wall time : {elapsed:.2f}s")
    print(f"  Avg/input : {avg_ms:.1f}ms")
    if elapsed:
        print(f"  Throughput: {len(inputs) / elapsed:.1f} inputs/s")

    if golds:
        acc = grade(preds, golds, verbose=args.verbose)
        print(
            f"\n  Accuracy (gold in top-3): {acc:.2%} ({int(acc * len(golds))}/{len(golds)})"
        )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
