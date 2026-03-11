# Scripts

Utility scripts for training data augmentation, prediction, and evaluation.

---

## Workflows

### Benchmark KenLM top-K accuracy

```bash
# 1. Generate top-K predictions from KenLM (e.g. top-64)
uv run python scripts/eval_kenlm_topk.py test \
    --work_dir work \
    --test_data data/madlad_multilang_clean_35k_optionB_kenlm/input_valid.txt \
    --test_output output/preds_char6_top64.txt \
    --k 64

# 2a. Grade at a single cutoff (e.g. top-3)
uv run python scripts/grade.py \
    --pred output/preds_char6_top64.txt \
    --answer data/madlad_multilang_clean_35k_optionB_kenlm/answer_valid.txt \
    --top-k 3

# 2b. Sweep K=1..64, compare models, save TSV
uv run python scripts/grade.py \
    --pred output/preds_char5_top64.txt output/preds_char6_top64.txt \
    --names char5 char6 \
    --answer data/madlad_multilang_clean_35k_optionB_kenlm/answer_valid.txt \
    --sweep --sweep-output output/sweep.tsv

# 3. Plot accuracy vs. K
#    Open scripts/plot_sweep.ipynb and run all cells
```

### Convert Lightning checkpoint to inference .pt

```bash
uv run python scripts/ckpt_to_pt.py work/best_reranker.ckpt work/reranker.pt
```

### Benchmark Gemini accuracy

```bash
# 1. Run Gemini on a set of inputs (produces a predictions file)
uv run python scripts/gemini_predictor.py \
    --input data/madlad_multilang_clean_35k_optionB_kenlm/input_valid.txt \
    --answer data/madlad_multilang_clean_35k_optionB_kenlm/answer_valid.txt \
    --output work/gemini_preds.txt

# 2. Grade (or use --eval-only to skip API calls on an existing file)
uv run python scripts/grade.py \
    --pred work/gemini_preds.txt \
    --answer data/madlad_multilang_clean_35k_optionB_kenlm/answer_valid.txt \
    --top-k 3
```

### Distillation pipeline (augment training data with Gemini predictions)

```bash
# 1. Export a W&B predictions table to get input.txt
uv run python scripts/parse_predictions_table.py \
    --table wandb_export.json \
    --input work/distill_input.txt

# 2. Run Gemini to get predictions for those inputs
uv run python scripts/gemini_predictor.py \
    --input work/distill_input.txt \
    --output work/distill_preds.txt

# 3. Build extended inputs (original prefix + each predicted char)
#    Outputs: work/distill_input_ext1.txt, work/distill_input_ext2.txt, work/distill_input_ext3.txt
uv run python scripts/extend_inputs.py \
    --input work/distill_input.txt \
    --pred work/distill_preds.txt \
    --outdir work/

# 4a. Append extended inputs to train.txt (produces a new augmented file)
uv run python scripts/augment_train.py \
    --train data/madlad_multilang_clean_35k_optionB_kenlm/train.txt \
    --ext work/distill_input_ext1.txt work/distill_input_ext2.txt work/distill_input_ext3.txt \
    --output data/madlad_multilang_clean_35k_optionB_kenlm/train_distillation.txt

# 4b. Or tokenize ext files into a standalone sequences file
uv run python scripts/prepare_distill_sequences.py \
    --ext work/distill_input_ext1.txt work/distill_input_ext2.txt work/distill_input_ext3.txt \
    --output data/distill/train.txt

# 5. (Optional) Build a validation split from distillation data
uv run python scripts/prepare_distill_valid.py \
    --input work/distill_input.txt \
    --preds work/distill_preds.txt \
    --outdir data/distill
```

---

## Script Reference

### `eval_kenlm_topk.py`

Generates KenLM top-K next-character predictions for a set of input prefixes. Outputs one prediction string per line with candidates in confidence order (e.g. `"etaio..."` for K=5). Pass the output to `grade.py` to evaluate accuracy at any cutoff.

### `grade.py`

Evaluates one or more predictions files against gold answers. A prediction is counted correct if the gold character appears within the first `--top-k` characters (case-insensitive, default top-3). With `--sweep`, prints an accuracy-vs-K table for every model side-by-side. Use `--names` to label each model (defaults to filename stems) and `--sweep-output` to save a TSV for plotting.

### `ckpt_to_pt.py`

Converts a Lightning `.ckpt` checkpoint to a plain inference `.pt` file containing `model_state_dict`, `tokens`, and `config`. The `.pt` format can be loaded at inference time without Lightning.

### `gemini_predictor.py`

Calls the Gemini API to predict the three most likely next characters for each input string. Supports async concurrency, exponential-backoff retries, a persistent prediction cache, and optional `--eval-only` mode to grade an existing predictions file without making API calls.

### `extend_inputs.py`

Takes an input file and a Gemini predictions file, then produces three output files where each line is the original input concatenated with the 1st, 2nd, or 3rd predicted character respectively. Used to generate augmented inputs for the distillation pipeline.

### `augment_train.py`

Copies an existing `train.txt` to `--output`, then appends tokenized extended-input lines from one or more `--ext` files. Each line is normalized and character-tokenized (matching `preprocess.py` format) before being written `--repeat` times.

### `prepare_distill_sequences.py`

Tokenizes extended-input files into a standalone sequences file in `train.txt` format. Use this when you want to keep distillation sequences as a separate file rather than merging them into an existing `train.txt`.

### `prepare_distill_valid.py`

Creates `input_valid.txt` and `answer_valid.txt` from distillation data. For each `(input, gemini_pred)` pair, emits three validation examples — one per predicted character.

### `parse_predictions_table.py`

Converts a Weights & Biases predictions table JSON export into flat `input.txt` and (optionally) `pred.txt` files, one entry per line. Useful for extracting model inputs/outputs logged during a W&B run.

### `plot_sweep.ipynb`

Jupyter notebook that reads one or more sweep TSVs produced by `grade.py --sweep --sweep-output` and plots accuracy vs. K on a log-scale x-axis using seaborn. Also prints a pivot table of accuracy at key cutoffs (K=1, 3, 5, 10, 20, 32, 64). Saves the plot to `output/sweep_plot.png`.

### `oov_test.py`

Diagnostic script that checks whether a given token is in the KenLM model vocabulary and prints its base score. Useful for debugging out-of-vocabulary behavior in the character-level language model.
