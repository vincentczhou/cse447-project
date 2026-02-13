# Data Pipeline

This directory contains scripts for building and preprocessing multilingual datasets for character-level language modeling with KenLM.

## Scripts Overview

### `builddataset.py`

Downloads and builds a multilingual dataset from the MADLAD-400 corpus.

**What it does:**
- Streams text data from [allenai/madlad-400](https://huggingface.co/datasets/allenai/madlad-400) for 53 languages
- Collects 1,000 samples per language (configurable)
- Filters texts by length (200-4,000 characters)
- Saves as a Hugging Face Dataset format
- Generates CSV summaries of language counts and rejected languages

**Output:**
- `data/madlad_multilang_clean_1k_optionB/` - Dataset directory
  - `dataset_dict.json` - Dataset metadata
  - `data-*.arrow` - Arrow-format data files
  - `language_counts.csv` - Per-language sample counts
  - `rejected_langs.csv` - Languages that failed to load (if any)

### `preprocess.py`

Prepares the downloaded dataset for KenLM training by converting to character-level tokenization.

**What it does:**
- Loads the dataset from `builddataset.py`
- Applies text normalization:
  - NFC unicode normalization (consistent canonical form)
  - Lowercase conversion
  - Whitespace normalization (collapse to single spaces)
- Character-level tokenization (spaces → `<sp>` token)
- Splits data into train/validation sets (99%/1%)
- Outputs plain text files suitable for KenLM

**Output:**
- `data/madlad_multilang_clean_1k_optionB_kenlm/` - Output directory
  - `train.txt` - Training data (one tokenized line per document)
  - `valid.txt` - Validation data

## Usage

### 1. Build the Dataset

```bash
cd src/data
uv run builddataset.py
```

This will take some time as it streams data from Hugging Face. Progress is printed per language.

**Note:** Requires `datasets==3.6.0` due to compatibility issues with newer versions.

### 2. Preprocess for KenLM

```bash
cd src/data
uv run preprocess.py
```

You'll see a progress bar showing processing status.

### 3. Train KenLM Model

Once preprocessing is complete, build a KenLM language model:

```bash
# Install KenLM first (if not already installed)
# See: https://github.com/kpu/kenlm

cd ../../data/madlad_multilang_clean_1k_optionB_kenlm

# Build a 5-gram model
lmplz -o 5 < train.txt > model_5gram.arpa

# Optionally, binarize for faster loading
build_binary model_5gram.arpa model_5gram.bin
```

## Configuration

All dataset and preprocessing settings live in the repo root:
- [data_config.yaml](../../data_config.yaml)

### `builddataset.py` Parameters

```python
TARGET_PER_LANG = 1000        # Samples to collect per language
MIN_CHARS = 200               # Minimum text length
MAX_CHARS = 4000              # Maximum text length (texts are truncated)
MAX_PER_LANG_SCAN = 200_000   # Max documents to scan per language
PATIENCE_PER_LANG = 50_000    # Early stop if no valid samples for N docs
SEED = 42                     # Random seed for reproducibility
```

To add/remove languages, edit the `LANGS` list.

### `preprocess.py` Parameters

```python
VALID_RATIO = 0.01                        # Fraction for validation split
SPACE_TOKEN = "<sp>"                      # Token to represent spaces
INPUT_DIR = "data/madlad_multilang_..."   # Input dataset path
OUTPUT_DIR = "data/madlad_multilang_..."  # Output directory
SEED = 42                                 # Random seed for train/val split
```

## Output Format

### Character Tokenization Example

Original text:
```
Hello world!
```

After preprocessing:
```
h e l l o <sp> w o r l d !
```

Each character becomes a space-separated token. Spaces in the original text are replaced with the `<sp>` token.

## Loading the Dataset Programmatically

```python
from datasets import Dataset

# Load the built dataset (paths are relative to repo root)
ds = Dataset.load_from_disk("data/madlad_multilang_clean_1k_optionB")
print(ds)
print(ds[0])  # First example with 'lang' and 'text' fields
```

## Dependencies

Required packages (already in `pyproject.toml`):
- `datasets==3.6.0` - Hugging Face datasets library
- `pandas` - CSV output
- `tqdm` - Progress bars

Install with:
```bash
uv sync
```

## Troubleshooting

### `trust_remote_code` Error

If you see errors about `trust_remote_code`, ensure you're using `datasets==3.6.0`. Newer versions (4.x) don't support the MADLAD-400 loading script.

```bash
uv add --dev "datasets==3.6.0"
```

### Memory Issues

The dataset is streamed, but preprocessing loads everything into memory. If you encounter memory issues:
- Reduce `TARGET_PER_LANG` in `builddataset.py`
- Process in batches by modifying `preprocess.py` to write incrementally

### Empty Dataset

If `builddataset.py` reports 0 samples:
- Check your internet connection
- Verify Hugging Face Hub access
- Check `rejected_langs.csv` for error messages
