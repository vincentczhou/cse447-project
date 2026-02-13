import sys
from collections import Counter
from datasets import Dataset
import json
from pathlib import Path
import random
from tqdm import tqdm
import yaml

# Add src/ to path so utils package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.text_utils import normalize_text, char_tokenize


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "data_config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR = BASE_DIR / config["data_dir"]
pre_cfg = config["preprocess"]

SEED = pre_cfg["seed"]
VALID_RATIO = pre_cfg["valid_ratio"]
FLUSH_INTERVAL = pre_cfg["flush_interval"]

INPUT_DIR = DATA_DIR / pre_cfg["input_dir"]
OUTPUT_DIR = DATA_DIR / pre_cfg["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ds = Dataset.load_from_disk(str(INPUT_DIR))
rng = random.Random(SEED)

train_path = OUTPUT_DIR / "train.txt"
valid_path = OUTPUT_DIR / "valid.txt"
valid_input_path = OUTPUT_DIR / "input_valid.txt"
valid_answer_path = OUTPUT_DIR / "answer_valid.txt"
vocab_path = OUTPUT_DIR / "vocab.json"

for path in (train_path, valid_path, valid_input_path, valid_answer_path, vocab_path):
    if path.exists():
        path.unlink()

train_count = 0
valid_count = 0
eval_count = 0
written_count = 0
vocab = Counter()

with (
    train_path.open("w", encoding="utf-8") as train_file,
    valid_path.open("w", encoding="utf-8") as valid_file,
    valid_input_path.open("w", encoding="utf-8") as input_valid_file,
    valid_answer_path.open("w", encoding="utf-8") as answer_valid_file,
):
    for ex in tqdm(ds, desc="Processing examples"):
        text = ex.get("text")
        if not text:
            continue

        norm = normalize_text(text)
        if not norm:
            continue

        tokenized = char_tokenize(norm)

        # Count token frequencies for vocabulary (after tokenization)
        tokens = tokenized.split()
        vocab.update(tokens)

        if rng.random() < VALID_RATIO:
            valid_file.write(tokenized + "\n")
            valid_count += 1
            written_count += 1

            # Create eval sample: pick a random split point in the normalized text
            # Use normalized text (no newlines, single-line safe)
            if len(norm) >= 2:
                # Random prefix length: at least 1 char, leave at least 1 for answer
                split_idx = rng.randint(1, len(norm) - 1)
                prefix = norm[:split_idx]
                answer = norm[split_idx]
                input_valid_file.write(prefix + "\n")
                answer_valid_file.write(answer + "\n")
                eval_count += 1
        else:
            train_file.write(tokenized + "\n")
            train_count += 1
            written_count += 1

        if written_count % FLUSH_INTERVAL == 0:
            train_file.flush()
            valid_file.flush()
            input_valid_file.flush()
            answer_valid_file.flush()

# Write vocabulary file as JSON (sorted by count descending)
vocab_dict = {token: count for token, count in vocab.most_common()}
with vocab_path.open("w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"Wrote {train_count} train lines to {train_path}")
print(f"Wrote {valid_count} valid lines to {valid_path}")
print(f"Wrote {eval_count} eval samples to {valid_input_path} + {valid_answer_path}")
print(f"Wrote {len(vocab)} unique characters to {vocab_path}")
