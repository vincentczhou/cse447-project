#!/usr/bin/env python
"""
Gemini API character-level next-character predictor (sequential version).

Sends one API request per input using structured JSON output with separate
fields for 1st, 2nd, and 3rd most likely next characters.
Processes inputs sequentially (no async/concurrency).

Usage:
    python src/scripts/gemini_predictor_sync.py \
        --input  example/input.txt \
        --output output/gemini_pred.txt \
        --answer example/answer.txt   # optional – prints accuracy
        --sample 100                  # optional – randomly sample N inputs
"""

import argparse
import json
import os
import random
import sys
import time

from google import genai
from google.genai import types

# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = (
    "You are a character prediction engine.\n"
    "Given a partial text, predict the three most likely next characters "
    "(including space and punctuation) in order of probability.\n"
    "Return a JSON object with exactly three keys:\n"
    '  "first"  – the single most likely next character\n'
    '  "second" – the second most likely next character\n'
    '  "third"  – the third most likely next character\n'
    "Each value must be exactly one character. "
    "Use a literal space character when space is your guess.\n"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "first": {"type": "STRING", "description": "Most likely next character"},
        "second": {
            "type": "STRING",
            "description": "Second most likely next character",
        },
        "third": {"type": "STRING", "description": "Third most likely next character"},
    },
    "required": ["first", "second", "third"],
}

FALLBACK_PRED = " ea"
CONTEXT_LIMIT = 200  # max chars of each input to include in prompt


def build_prompt(text: str) -> str:
    """Build a prompt for a single input.
    Long inputs are trimmed to the last CONTEXT_LIMIT chars."""
    trimmed = text if len(text) <= CONTEXT_LIMIT else "…" + text[-CONTEXT_LIMIT:]
    return f'Predict the next character for this partial text:\n"{trimmed}"'


def parse_response(raw: str) -> str:
    """Parse a structured JSON response into a 3-char prediction."""
    try:
        data = json.loads(raw)
        first = str(data.get("first", " "))[:1] or " "
        second = str(data.get("second", "e"))[:1] or "e"
        third = str(data.get("third", "a"))[:1] or "a"
        return first + second + third
    except (json.JSONDecodeError, TypeError, AttributeError):
        return FALLBACK_PRED


def predict_single(
    client: genai.Client,
    text: str,
    config: types.GenerateContentConfig,
) -> tuple[str, float]:
    """Send one synchronous API request for a single input."""
    prompt = build_prompt(text)

    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        raw = resp.text if resp.text else ""
    except Exception as e:
        print(f"  [error] API call failed: {e}", file=sys.stderr)
        raw = ""
    elapsed = time.perf_counter() - t0

    return parse_response(raw), elapsed


def predict_all(client: genai.Client, inputs: list[str]) -> tuple[list[str], float]:
    """Predict all inputs sequentially, one at a time."""
    print(f"  {len(inputs)} inputs → {len(inputs)} API call(s) (sequential)")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=64,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    all_preds: list[str] = []
    total_time = 0.0
    for i, text in enumerate(inputs):
        pred, elapsed = predict_single(client, text, config)
        all_preds.append(pred)
        total_time += elapsed
        if (i + 1) % 50 == 0 or i == len(inputs) - 1:
            print(f"  [{i + 1}/{len(inputs)}] {total_time:.1f}s elapsed", flush=True)

    return all_preds, total_time


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def grade(preds: list[str], golds: list[str], verbose: bool = False) -> float:
    correct = 0
    for i, (p, g) in enumerate(zip(preds, golds)):
        hit = g.lower() in p[:3].lower()
        correct += hit
        if verbose:
            tag = "✓" if hit else "✗"
            print(f"  {tag} input {i}: gold='{g}' pred='{p}'")
    return correct / len(golds) if golds else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Gemini character-level predictor (sequential)"
    )
    parser.add_argument("--input", required=True, help="Path to input.txt")
    parser.add_argument("--output", default="output/gemini_pred.txt")
    parser.add_argument("--answer", default=None, help="Path to answer.txt (optional)")
    parser.add_argument("--api-key", default=None, help="Gemini API key")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N inputs (for benchmarking large datasets)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    key = args.api_key or API_KEY
    if not key:
        sys.exit("Error: set GEMINI_API_KEY env var or pass --api-key")

    client = genai.Client(api_key=key)

    print(f"Model : {MODEL_NAME}")
    inputs = load_lines(args.input)
    golds = load_lines(args.answer) if args.answer else None

    # Optionally sample
    if args.sample and args.sample < len(inputs):
        random.seed(42)
        indices = sorted(random.sample(range(len(inputs)), args.sample))
        inputs = [inputs[i] for i in indices]
        if golds:
            golds = [golds[i] for i in indices]
        print(
            f"Sampled {args.sample} of {len(load_lines(args.input))} inputs (seed=42)"
        )

    print(f"Inputs: {len(inputs)} lines\n")

    # ── Predict (sequential, 1 API call per input) ─────────────────────────
    preds, elapsed = predict_all(client, inputs)

    # ── Write predictions ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"\nPredictions written to {args.output}")

    # ── Report latency ─────────────────────────────────────────────────────
    avg_ms = (elapsed / len(inputs)) * 1000 if inputs else 0
    print(f"\n{'=' * 50}")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Inputs        : {len(inputs)}")
    print(f"  Total latency : {elapsed:.2f}s")
    print(f"  Per-input avg : {avg_ms:.1f}ms")
    if elapsed:
        print(f"  Throughput    : {len(inputs) / elapsed:.1f} inputs/s")

    # ── Grade (optional) ───────────────────────────────────────────────────
    if golds:
        print(f"\n  Grading against {args.answer} ({len(golds)} answers)")
        acc = grade(preds, golds, verbose=args.verbose)
        print(
            f"\n  Accuracy (gold in top-3): {acc:.2%}  ({int(acc * len(golds))}/{len(golds)})"
        )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
