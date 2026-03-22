# Reranker

Two-stage next-character prediction: KenLM generates top-K candidates, then a Transformer reranker rescores them to pick the best prediction.

## Package Structure

| File | Purpose |
|------|---------|
| `config.py` | `RerankerConfig` (architecture) and `TrainConfig` (training) dataclasses, YAML loading |
| `model.py` | `Reranker` nn.Module — causal Transformer with KenLM score blending via mixture model |
| `dataset.py` | `RerankerDataset` (random negatives) and `PrecomputedRerankerDataset` (KenLM hard negatives), collation functions |
| `lightning_module.py` | `RerankerLightningModule` — training/validation steps, optimizer/scheduler config |
| `train.py` | Training entry point — loads config, data, runs Lightning Trainer |

## Training Workflow

### 1. Prepare data

You need tokenized sequence files (`train.txt`, `valid.txt`) and a vocabulary JSON. Optionally, precompute KenLM candidate TSVs for hard-negative training (see [src/data/README.md](../data/README.md)).

### 2. Configure `config.yaml`

All hyperparameters live in the `reranker` section of `config.yaml` at the repo root:

- `reranker.architecture` — model dimensions, attention heads, layers, dropout, temperature, alpha
- `reranker.training` — batch size, learning rate, warmup, epochs, scheduler
- `reranker.data` — paths to train/valid sequences, vocab, and optional candidate TSVs
- `reranker.early_stopping` — patience, metric, mode
- `reranker.output` — checkpoint paths, save frequency

### 3. Train

```bash
uv run python src/reranker/train.py
```

Training reads all config from `config.yaml` (no CLI arguments). Two checkpoint files are produced:

- `work/best_reranker.ckpt` — Lightning checkpoint saved by the `ModelCheckpoint` callback whenever validation metric improves (or every `every_n_train_steps` steps if configured).
- `work/<checkpoint_name>.pt` — plain inference checkpoint (no Lightning dependency) written unconditionally at the end of the training run, including on Ctrl+C interruption. This is the file used by `src/predict.py`.

### 4. Convert checkpoint (if needed)

If you only have a Lightning `.ckpt` and need an inference `.pt`:

```bash
uv run python scripts/ckpt_to_pt.py work/best_reranker.ckpt work/reranker.pt
```

## Inference

```bash
uv run python src/predict.py --work_dir work --test_data example/input.txt --test_output pred.txt
```

Falls back to KenLM-only if no reranker checkpoint is found in `work_dir`. Use `--kenlm-only` to force KenLM-only mode. Use `--alpha` to override the KenLM blending weight at inference time.

## Dataset Modes

**`RerankerDataset`** (random negatives): Built from raw tokenized sequences. Each example is a `(context, target)` pair. Negatives are sampled from unigram frequency distribution at collation time. Used when no precomputed candidate TSVs are configured.

**`PrecomputedRerankerDataset`** (KenLM hard negatives): Backed by a precomputed TSV from `precompute_kenlm_candidates.py`. Each row contains KenLM's top-K candidates and their scores. Supports two loading modes:
- `lazy_load_candidates: true` — byte-offset indexing, low RAM usage for large TSVs
- `lazy_load_candidates: false` — loads all rows into memory, faster `__getitem__`

## How `max_examples` and Lazy Loading Work

Both dataset classes support a `max_examples` cap to limit dataset size without loading everything.

### `RerankerDataset` (random negatives)

Every `(seq_idx, pos)` pair in the loaded sequences is a potential example. With `max_examples=None`, the full flat index is materialized. With `max_examples` set, a memory-efficient cumsum/bisect approach is used:

1. **pos_counts**: for each sequence, count valid positions (`len(seq) - 1`). This is just a list of small integers.
2. **cumsum**: prefix sum over pos_counts via `itertools.accumulate`. Maps flat indices to sequences. E.g. `pos_counts = [3, 5, 2]` → `cumsum = [3, 8, 10]`, so flat indices 0-2 belong to seq 0, 3-7 to seq 1, 8-9 to seq 2.
3. **random.sample**: picks `max_examples` distinct flat indices from `range(total_positions)` — Python handles range objects in O(k) memory, not O(N).
4. **bisect_right**: converts each flat index back to `(seq_idx, pos)` — `bisect_right(cumsum, f)` gives the sequence, and the remainder gives the position within that sequence (offset by +1 since positions start at 1).

This avoids materializing millions of `(seq_idx, pos)` tuples when you only need a subset.

### `PrecomputedRerankerDataset` (KenLM hard negatives)

Two modes controlled by `lazy`:

**`lazy=True`** (byte-offset indexing):
- `__init__` does one binary-mode pass over the TSV, recording the byte offset of each valid row (~8 bytes per offset).
- If `max_examples` is set, offsets are shuffled and truncated.
- `__getitem__` seeks to the byte offset and parses only that single row on demand.
- Peak memory: proportional to `num_rows × 8 bytes` (offsets only), not the full parsed TSV.
- Trade-off: each `__getitem__` does a disk seek + readline, so throughput is lower than eager mode.

**`lazy=False`** (eager, in-memory):
- `__init__` parses all TSV rows into memory as `(seq_idx, pos, cand_ids, kenlm_scores, label)` tuples.
- If `max_examples` is set, examples are shuffled and truncated.
- `__getitem__` is a simple list index lookup — no disk I/O.
- Trade-off: higher RAM usage, but faster training throughput.

In both cases, `__getitem__` looks up the original sequence via `self.sequences[seq_idx]` to extract the context window `seq[max(0, pos - max_context_len) : pos]`.
