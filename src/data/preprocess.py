from datasets import Dataset
from pathlib import Path
import random
import re
import unicodedata
from tqdm import tqdm

SEED = 42
VALID_RATIO = 0.01
SPACE_TOKEN = "<sp>"

INPUT_DIR = Path("data/madlad_multilang_clean_1k_optionB")
OUTPUT_DIR = Path("data/madlad_multilang_clean_1k_optionB_kenlm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ws_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _ws_re.sub(" ", text).strip()
    return text


def char_tokenize(text: str) -> str:
    chars = [SPACE_TOKEN if ch == " " else ch for ch in text]
    return " ".join(chars)


ds = Dataset.load_from_disk(str(INPUT_DIR))
rng = random.Random(SEED)

train_path = OUTPUT_DIR / "train.txt"
valid_path = OUTPUT_DIR / "valid.txt"

train_lines = []
valid_lines = []

for ex in tqdm(ds, desc="Processing examples"):
    text = ex.get("text")
    if not text:
        continue

    norm = normalize_text(text)
    if not norm:
        continue

    tokenized = char_tokenize(norm)
    if rng.random() < VALID_RATIO:
        valid_lines.append(tokenized)
    else:
        train_lines.append(tokenized)

with train_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(train_lines) + "\n")

with valid_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(valid_lines) + "\n")

print(f"Wrote {len(train_lines)} train lines to {train_path}")
print(f"Wrote {len(valid_lines)} valid lines to {valid_path}")
