from datasets import load_dataset, Dataset, Features, Value
from collections import defaultdict
import random
import os
import pandas as pd

SEED = 42
random.seed(SEED)

TARGET_PER_LANG = 1000
MIN_CHARS = 200
MAX_CHARS = 4000

MAX_PER_LANG_SCAN = 200_000   # hard cap per language
PATIENCE_PER_LANG = 50_000    # stop if no adds for this many scanned docs

LANGS = [
    # Western + Central European
    "en", "es", "fr", "de", "pt", "it", "nl", "sv", "da", "no", "fi", "pl", "cs", "sk", "hu", "ro",
    "bg", "sr", "hr", "sl", "lt", "lv", "et", "el", "uk",
    # Common global
    "ru", "ar", "he", "fa", "tr",
    # South/Southeast Asia
    "hi", "bn", "ur", "ta", "te", "mr", "gu", "pa", "ne", "si", "th", "vi", "id", "ms",
    # East Asia
    "zh", "ja", "ko",
    # Africa (higher-resource)
    "sw", "ha", "yo", "ig", "am", "so"
]

features = Features({"lang": Value("string"), "text": Value("string")})

samples = []
counts = defaultdict(int)
rejected = []  # unsupported language codes or load failures

for lang in LANGS:
    print(f"\n=== {lang} ===")
    try:
        ds = load_dataset(
            "allenai/madlad-400",
            lang,
            split="clean",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        rejected.append({"lang": lang, "reason": str(e).splitlines()[-1]})
        print(f"Skipping {lang}: {rejected[-1]['reason']}")
        continue

    scanned = 0
    since_last_add = 0

    for ex in ds:
        scanned += 1
        since_last_add += 1

        if scanned >= MAX_PER_LANG_SCAN:
            print(f"Reached MAX_PER_LANG_SCAN={MAX_PER_LANG_SCAN}.")
            break
        if since_last_add >= PATIENCE_PER_LANG:
            print(f"No adds for PATIENCE_PER_LANG={PATIENCE_PER_LANG}.")
            break
        if counts[lang] >= TARGET_PER_LANG:
            break

        text = ex.get("text") or ex.get("document") or ex.get("content")
        if not text or len(text) < MIN_CHARS:
            continue

        text = text[:MAX_CHARS]
        samples.append({"lang": lang, "text": text})
        counts[lang] += 1
        since_last_add = 0

    print(f"Collected {counts[lang]} / {TARGET_PER_LANG} (scanned {scanned})")

print("\nCounts summary:", dict(counts))
print("Total samples:", len(samples))

out_dir = "data/madlad_multilang_clean_1k_optionB"
os.makedirs(out_dir, exist_ok=True)

if len(samples) == 0:
    raise RuntimeError(
        "Collected 0 samples total — check dataset access/fields.")

ds_out = Dataset.from_list(samples, features=features)
ds_out.save_to_disk(out_dir)

pd.DataFrame([{"lang": l, "count": counts[l]} for l in LANGS]).to_csv(
    os.path.join(out_dir, "language_counts.csv"), index=False
)

if rejected:
    pd.DataFrame(rejected).to_csv(os.path.join(
        out_dir, "rejected_langs.csv"), index=False)

print("Saved to:", out_dir)
