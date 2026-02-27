from datasets import load_dataset, Dataset, Features, Value
from collections import defaultdict
from pathlib import Path
import random
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "data_config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR = BASE_DIR / config["data_dir"]
build_cfg = config["builddataset"]

SEED = build_cfg["seed"]
random.seed(SEED)

TARGET_PER_LANG = build_cfg["target_per_lang"]
MIN_CHARS = build_cfg["min_chars"]
MAX_CHARS = build_cfg["max_chars"]

MAX_PER_LANG_SCAN = build_cfg["max_per_lang_scan"]
PATIENCE_PER_LANG = build_cfg["patience_per_lang"]

LANGS = build_cfg["langs"]

SHARD_ROWS = build_cfg.get("shard_rows", 20000)  # adjust if needed
shard_idx = 0
buffer = []

out_dir = DATA_DIR / build_cfg["output_dir"]
out_dir.mkdir(parents=True, exist_ok=True)

features = Features({"lang": Value("string"), "text": Value("string")})

counts = defaultdict(int)
rejected = []  # unsupported language codes or load failures


def flush_shard():
    global shard_idx, buffer
    if not buffer:
        return
    ds_shard = Dataset.from_list(buffer, features=features)
    shard_path = out_dir / "parts" / f"part-{shard_idx:05d}.parquet"
    ds_shard.to_parquet(str(shard_path))
    shard_idx += 1
    buffer = []


for lang in LANGS:
    print(f"\n=== {lang} ===")
    try:
        ds = load_dataset(
            "allenai/madlad-400",
            lang,
            split="clean",
            streaming=True,
            trust_remote_code=True,
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
        buffer.append({"lang": lang, "text": text})
        counts[lang] += 1
        since_last_add = 0
        if len(buffer) >= SHARD_ROWS:
            flush_shard()
    print(f"Collected {counts[lang]} / {TARGET_PER_LANG} (scanned {scanned})")
flush_shard()
print("\nCounts summary:", dict(counts))
print("Total samples:", sum(counts.values()))


if sum(counts.values()) == 0:
    raise RuntimeError("Collected 0 samples total — check dataset access/fields.")

parquet_files = sorted(str(p) for p in (out_dir / "parts").glob("part-*.parquet"))
ds_out = load_dataset("parquet", data_files=parquet_files)
ds_out.save_to_disk(str(out_dir))

pd.DataFrame([{"lang": lang, "count": counts[lang]} for lang in LANGS]).to_csv(
    out_dir / "language_counts.csv", index=False
)

if rejected:
    pd.DataFrame(rejected).to_csv(out_dir / "rejected_langs.csv", index=False)

print("Saved to:", out_dir)
