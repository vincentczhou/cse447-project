#!/usr/bin/env python
"""
Gemini API character-level next-character predictor.

Batches inputs into chunked API calls (~50 per call to stay within token limits),
parses multi-line responses, then reports latency and accuracy.

Usage:
    python src/gemini_predictor.py \
        --input  example/input.txt \
        --output output/gemini_pred.txt \
        --answer example/answer.txt   # optional – prints accuracy
        --sample 100                  # optional – only use first N inputs
"""

import argparse
import math
import os
import random
import sys
import time

from google import genai
from google.genai import types

# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a character prediction engine.\n"
    "You will receive a numbered list of partial texts.\n"
    "For EACH partial text, predict the 3 most likely next characters "
    "(including space and punctuation) in order from most to least likely.\n\n"
    "Return a JSON array of strings, one per input, in the same order.\n"
    "Each string must contain EXACTLY 3 characters (your top-3 guesses).\n"
    "Use a literal space character when space is one of your guesses.\n\n"
    "Example for 2 inputs: [\"y N\", \"e a\"]\n"
)

FALLBACK_PRED = " ea"
CHUNK_SIZE = 50  # inputs per API call – balances token limits vs. request count
CONTEXT_LIMIT = 200  # max chars of each input to include in prompt (trim long contexts)


def build_batch_prompt(inputs: list[str]) -> str:
    """Build a single prompt containing all inputs as a numbered list.
    Long inputs are trimmed to the last CONTEXT_LIMIT chars (the ending matters most)."""
    lines = []
    for i, inp in enumerate(inputs, 1):
        trimmed = inp if len(inp) <= CONTEXT_LIMIT else "…" + inp[-CONTEXT_LIMIT:]
        lines.append(f'{i}. "{trimmed}"')
    return "Predict next 3 characters for each partial text:\n" + "\n".join(lines)


def parse_batch_response(raw: str, expected: int) -> list[str]:
    """Parse the model's JSON array response into per-input 3-char predictions."""
    import json
    preds: list[str] = []
    try:
        # Try JSON parse first (structured output)
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                s = str(item)
                pred = (s + "   ")[:3]
                preds.append(pred)
    except (json.JSONDecodeError, TypeError):
        # Fallback to line-by-line parsing
        lines = [l for l in raw.split("\n") if l.strip()]
        for line in lines:
            cleaned = line
            if len(cleaned) > 2 and cleaned[0].isdigit():
                for sep in [". ", ": ", ") ", " "]:
                    idx = cleaned.find(sep)
                    if idx != -1 and idx <= 3 and cleaned[:idx].isdigit():
                        cleaned = cleaned[idx + len(sep):]
                        break
            cleaned = cleaned.strip().strip('"').strip("'").strip("`")
            pred = (cleaned + "   ")[:3]
            preds.append(pred)

    # Pad with fallback if model returned fewer lines
    while len(preds) < expected:
        preds.append(FALLBACK_PRED)
    return preds[:expected]


def predict_chunk(client: genai.Client, inputs: list[str]) -> tuple[list[str], float]:
    """Send one chunk of inputs in a single API call."""
    prompt = build_batch_prompt(inputs)

    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=max(512, len(inputs) * 15),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            ),
        )
        raw = resp.text if resp.text else ""
    except Exception as e:
        print(f"  [error] API call failed: {e}", file=sys.stderr)
        raw = ""
    elapsed = time.perf_counter() - t0
    preds = parse_batch_response(raw, len(inputs))
    return preds, elapsed


def predict_all(client: genai.Client, inputs: list[str]) -> tuple[list[str], float]:
    """Predict in chunks to handle large datasets."""
    n_chunks = math.ceil(len(inputs) / CHUNK_SIZE)
    print(f"  {len(inputs)} inputs → {n_chunks} API call(s) (chunk size {CHUNK_SIZE})")

    all_preds: list[str] = []
    total_time = 0.0
    for ci in range(n_chunks):
        start = ci * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(inputs))
        chunk = inputs[start:end]
        print(f"  Chunk {ci+1}/{n_chunks} ({len(chunk)} inputs) …", end=" ", flush=True)
        preds, elapsed = predict_chunk(client, chunk)
        all_preds.extend(preds)
        total_time += elapsed
        print(f"done in {elapsed:.2f}s")
        # Small delay between chunks to avoid rate limits
        if ci < n_chunks - 1:
            time.sleep(1)

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
    parser = argparse.ArgumentParser(description="Gemini character-level predictor")
    parser.add_argument("--input", required=True, help="Path to input.txt")
    parser.add_argument("--output", default="output/gemini_pred.txt")
    parser.add_argument("--answer", default=None, help="Path to answer.txt (optional)")
    parser.add_argument("--api-key", default=None, help="Gemini API key")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N inputs (for benchmarking large datasets)")
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
        print(f"Sampled {args.sample} of {len(load_lines(args.input))} inputs (seed=42)")

    print(f"Inputs: {len(inputs)} lines\n")

    # ── Predict ────────────────────────────────────────────────────────────
    preds, elapsed = predict_all(client, inputs)

    # ── Write predictions ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"\nPredictions written to {args.output}")

    # ── Report latency ─────────────────────────────────────────────────────
    avg_ms = (elapsed / len(inputs)) * 1000 if inputs else 0
    print(f"\n{'='*50}")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Inputs        : {len(inputs)}")
    print(f"  Total latency : {elapsed:.2f}s")
    print(f"  Per-input avg : {avg_ms:.1f}ms")
    if elapsed:
        print(f"  Throughput    : {len(inputs)/elapsed:.1f} inputs/s")

    # ── Grade (optional) ───────────────────────────────────────────────────
    if golds:
        print(f"\n  Grading against {args.answer} ({len(golds)} answers)")
        acc = grade(preds, golds, verbose=args.verbose)
        print(f"\n  Accuracy (gold in top-3): {acc:.2%}  ({int(acc*len(golds))}/{len(golds)})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
