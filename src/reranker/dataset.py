"""Data loading, datasets, and collation functions for the reranker.

Provides:
- load_vocab / load_sequences: file loading utilities
- RerankerDataset: random-negative dataset built from raw sequences
- PrecomputedRerankerDataset: hard-negative dataset backed by a precomputed KenLM TSV
- collate_reranker / collate_precomputed: collation functions for DataLoader
"""

from __future__ import annotations

import bisect
import json
import random
from itertools import accumulate, islice
from pathlib import Path

import torch
from tqdm import tqdm

from .config import MISSING_LABEL, PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN


# ---------------------------------------------------------------------------
# Chunked data-loading helpers
# ---------------------------------------------------------------------------

LOAD_CHUNK_SIZE = 10_000  # lines per chunk for streaming file reads


def _iter_line_chunks(f, chunk_size: int = LOAD_CHUNK_SIZE):
    """Yield successive chunks of lines from a file handle, reading lazily."""
    while True:
        chunk = list(islice(f, chunk_size))
        if not chunk:
            break
        yield chunk


def _parse_tsv_chunk(
    lines: list[str],
    stoi: dict[str, int],
    n_sequences: int,
) -> list[tuple[int, int, list[int], list[float], int]]:
    """Parse a chunk of precomputed-candidate TSV lines."""
    examples = []
    for line in lines:
        # Skip rows where gold token is \x01 (the intra-field separator),
        # which corrupts the candidate split.
        if line.rstrip("\n").endswith("\t\x01"):
            continue
        seq_idx_s, pos_s, cands_s, scores_s, gold = line.rstrip("\n").split(
            "\t", maxsplit=4
        )
        seq_idx = int(seq_idx_s)
        # Guard: TSV may cover more lines than were loaded (e.g. max_train_lines)
        if seq_idx >= n_sequences:
            continue
        cand_tokens = cands_s.split("\x01")
        cand_ids = [stoi.get(t, UNK_ID) for t in cand_tokens]
        kenlm_scores = [float(s) for s in scores_s.split("\x01")]
        try:
            label = cand_tokens.index(gold)
        except ValueError:
            print(
                f"WARNING: gold token {gold!r} not found in candidates at "
                f"seq_idx={seq_idx_s}, pos={pos_s}. Marking this row as missing."
            )
            label = MISSING_LABEL
        examples.append((seq_idx, int(pos_s), cand_ids, kenlm_scores, label))
    return examples


def _parse_seq_chunk(lines: list[str], stoi: dict[str, int]) -> list[list[int]]:
    """Parse a chunk of tokenized text lines into lists of token IDs.

    Returns only sequences with >=2 tokens (need >=1 context + 1 target).
    """
    sequences: list[list[int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ids = [stoi.get(tok, UNK_ID) for tok in line.split()]
        if len(ids) >= 2:
            sequences.append(ids)
    return sequences


# ---------------------------------------------------------------------------
# Vocabulary & sequence loading
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: Path) -> tuple[list[str], torch.Tensor]:
    """Load vocab.json, prepend <pad> and <unk>, return tokens and unigram probs.

    Returns:
        tokens: list of token strings indexed by ID.
        unigram_probs: [V] tensor of normalized probabilities (pad=0).
    """
    # Unigram probs are used as sampling weights for random negatives in
    # collate_reranker(). When training with precomputed KenLM candidates,
    # negatives come from the TSV and these probs are unused.
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_counts: dict[str, int] = json.load(f)

    tokens = [PAD_TOKEN, UNK_TOKEN]
    counts = [0.0, 1.0]

    for token, count in vocab_counts.items():
        if token in (PAD_TOKEN, UNK_TOKEN):
            continue
        tokens.append(token)
        counts.append(float(max(1, int(count))))

    probs = torch.tensor(counts, dtype=torch.float)
    probs[PAD_ID] = 0.0
    probs = probs / probs.sum()
    return tokens, probs


def load_sequences(
    path: Path,
    stoi: dict[str, int],
    max_lines: int | None = None,
) -> list[torch.Tensor]:
    """Load tokenized text file into a list of 1-D LongTensors.

    Streams the file lazily in chunks to avoid loading it all into memory.
    """
    sequences: list[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as f:
        line_iter = islice(f, max_lines) if max_lines is not None else f
        for chunk in tqdm(
            _iter_line_chunks(line_iter),
            desc=f"Loading {path.name}",
        ):
            for ids in _parse_seq_chunk(chunk, stoi):
                sequences.append(torch.tensor(ids, dtype=torch.long))
    return sequences


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------


class RerankerDataset(torch.utils.data.Dataset):
    """Lazy dataset that yields (context_ids, target_id) pairs.

    Instead of materializing millions of Example objects up front, we store a
    flat index of (sequence_index, position) pairs.  __getitem__ slices into
    the raw sequences on the fly.

    Each valid position `pos` in a sequence produces one example:
        context = seq[max(0, pos - max_context_len) : pos]   (variable length)
        target  = seq[pos]                                    (single token ID)

    Padding is deferred to collate_reranker().
    """

    def __init__(
        self,
        sequences: list[torch.Tensor],
        max_context_len: int,
        max_examples: int | None = None,
    ):
        self.sequences = sequences
        self.max_context_len = max_context_len

        # Build flat index: each entry is (seq_idx, position).
        # Position starts at 1 (need at least 1 context token before target).
        if max_examples is None:
            # No cap: materialise the full index straight away.
            self.index: list[tuple[int, int]] = []
            for seq_idx, seq in enumerate(sequences):
                for pos in range(1, len(seq)):
                    self.index.append((seq_idx, pos))
        else:
            # Memory-efficient random sampling without building the full index.
            #
            # Conceptually, every (seq_idx, pos) pair has a unique "flat index"
            # in [0, total_positions). We want to pick max_examples of those at
            # random and convert them back to (seq_idx, pos) pairs.
            #
            # Step 1 — pos_counts[i] = number of valid positions in sequence i
            #   (i.e. len(seq) - 1, because pos runs from 1 to len-1 inclusive).
            #   This is just a list of small integers, negligible memory.
            #
            # Step 2 — cumsum[i] = total positions in sequences 0..i (inclusive),
            #   built with itertools.accumulate. Example:
            #     pos_counts = [3, 5, 2]  →  cumsum = [3, 8, 10]
            #   Flat indices 0-2  belong to seq 0 (positions 1,2,3).
            #   Flat indices 3-7  belong to seq 1 (positions 1,2,3,4,5).
            #   Flat indices 8-9  belong to seq 2 (positions 1,2).
            #
            # Step 3 — random.sample(range(total_positions), max_examples) picks
            #   exactly max_examples distinct flat indices without materialising
            #   the full range (Python's random.sample handles range objects
            #   efficiently in O(max_examples) time and space).
            #
            # Step 4 — for each sampled flat index f:
            #   • bisect_right(cumsum, f) gives the sequence index i.
            #   • The position within that sequence is:
            #       pos = f - cumsum[i-1] + 1  (or just f + 1 when i == 0,
            #       because cumsum[-1] would be 0 for the boundary before seq 0).
            #     The +1 accounts for the fact that position 0 is reserved for
            #     the first context token — valid positions start at 1.
            pos_counts = [max(0, len(seq) - 1) for seq in sequences]
            cumsum = list(accumulate(pos_counts))
            total_positions = cumsum[-1] if cumsum else 0
            n = min(max_examples, total_positions)
            sampled_flat = sorted(random.sample(range(total_positions), n))
            self.index = []
            for f in sampled_flat:
                i = bisect.bisect_right(cumsum, f)
                prev = cumsum[i - 1] if i > 0 else 0
                pos = f - prev + 1
                self.index.append((i, pos))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq_idx, pos = self.index[idx]
        seq = self.sequences[seq_idx]

        # Slice context (variable length, no padding yet)
        ctx_start = max(0, pos - self.max_context_len)
        context = seq[ctx_start:pos]  # 1-D LongTensor, length 1..max_context_len
        target = seq[pos].item()  # single integer

        return context, target


def collate_reranker(
    batch: list[tuple[torch.Tensor, int]],
    candidate_size: int,
    unigram_probs: torch.Tensor,
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a batch of (context, target) pairs into padded tensors with
    vectorized negative sampling.

    For each example, builds a candidate set of size M:
        - 1 gold target (placed at a random position)
        - M-1 negative samples drawn from unigram_probs

    Args:
        batch: list of (context_ids: [T_i], target_id: int) from RerankerDataset.
        candidate_size: M, total number of candidates per example.
        unigram_probs: [V] tensor of sampling probabilities for negatives.
        pad_id: token ID used for right-padding contexts.

    Returns:
        context_ids:   [B, T_max] right-padded context token IDs.
        candidate_ids: [B, M] candidate token IDs (gold is hidden among negatives).
        kenlm_scores:  [B, M] zeros (KenLM scores unavailable for random negatives).
        labels:        [B] index of the gold target within each candidate set.
    """
    contexts, targets = zip(*batch)
    B = len(contexts)
    M = candidate_size

    # --- Right-pad contexts to the longest in the batch ---
    context_ids = torch.nn.utils.rnn.pad_sequence(
        contexts, batch_first=True, padding_value=pad_id
    )  # [B, T_max]

    # --- Vectorized negative sampling ---
    # Sample M candidates per example from unigram distribution
    candidate_ids = torch.multinomial(
        unigram_probs.unsqueeze(0).expand(B, -1),
        num_samples=M,
        replacement=False,
    )  # [B, M]

    # Place the gold target at a random position in each candidate set
    gold = torch.tensor(targets, dtype=torch.long)  # [B]
    labels = torch.randint(0, M, (B,))  # [B] — random position for gold
    candidate_ids[torch.arange(B), labels] = gold

    # kenlm_scores are zeros — not available for random negatives.
    # Shape matches precomputed collate so training_step has a uniform interface.
    kenlm_scores = torch.zeros(B, M, dtype=torch.float)

    return context_ids, candidate_ids, kenlm_scores, labels


# ---------------------------------------------------------------------------
# Precomputed-candidate dataset & collation
# ---------------------------------------------------------------------------


class PrecomputedRerankerDataset(torch.utils.data.Dataset):
    """Dataset backed by a precomputed KenLM top-K candidate TSV.

    Each TSV row is one training example. Candidates are the KenLM top-K tokens
    for that (seq_idx, pos) — hard negatives the reranker must distinguish from
    gold. This is strictly better than random frequency-weighted negatives because
    the model sees realistic confusable characters.

    The candidate set size M is fixed by the TSV's --k flag. The `candidate_size`
    field in TrainConfig is ignored when this dataset is active.

    TSV format (tab-separated; candidates/scores use \\x01 as intra-field sep):
        seq_idx  pos  candidates  kenlm_scores  gold

    Supports two loading strategies controlled by the ``lazy`` flag:

    - **lazy=True** (default): __init__ does a single fast pass in binary mode
      to record the byte offset of each valid row (~8 bytes per row).
      __getitem__ seeks to that offset and parses only that one row on demand.
      Peak memory is proportional to the number of rows × 8 bytes, not the
      full parsed file contents.  Best for very large TSVs that don't fit in RAM.

    - **lazy=False**: All rows are parsed up front into in-memory lists
      (via _parse_tsv_chunk).  __getitem__ is a simple index lookup with no disk
      I/O, so training throughput is higher.  Use this when the TSV fits
      comfortably in RAM.

    Args:
        tsv_path: Path to the precomputed TSV.
        sequences: Token-ID tensors from load_sequences(), indexed by seq_idx.
        stoi: Token-string → ID mapping.
        max_context_len: Context window; context = seq[max(0, pos-L) : pos].
        max_examples: Cap dataset size; rows/offsets are randomly shuffled then truncated.
        lazy: If True, use byte-offset lazy loading; if False, load all rows into memory.
    """

    def __init__(
        self,
        tsv_path: Path,
        sequences: list[torch.Tensor],
        stoi: dict[str, int],
        max_context_len: int,
        max_examples: int | None = None,
        lazy: bool = True,
    ):
        self.sequences = sequences
        self.max_context_len = max_context_len
        self.tsv_path = tsv_path
        self.stoi = stoi
        self.lazy = lazy
        # File handle opened lazily in __getitem__ (once per DataLoader worker).
        self._fh: object = None

        n_sequences = len(sequences)

        if lazy:
            print(f"Loading {tsv_path.name} (lazy: byte-offset indexing)")
            # Index pass: open in binary mode so f.tell() gives reliable byte offsets.
            offsets: list[int] = []
            skipped = 0
            with tsv_path.open("rb") as f:
                next(f)  # skip header
                pbar = tqdm(desc=f"Indexing {tsv_path.name}", unit=" rows")
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    tab = line.index(b"\t")
                    if int(line[:tab]) < n_sequences:
                        if line.rstrip(b"\n").endswith(b"\t\x01"):
                            skipped += 1
                        else:
                            offsets.append(offset)
                    pbar.update(1)
                pbar.close()
            if skipped:
                print(f"Skipped {skipped} rows with gold=\\x01")

            if max_examples is not None and len(offsets) > max_examples:
                random.shuffle(offsets)
                offsets = offsets[:max_examples]

            self.offsets = offsets
            self.examples: list[tuple[int, int, list[int], list[float], int]] | None = (
                None
            )
            print(f"Indexed {len(offsets):,} rows from {tsv_path.name}")
        else:
            print(f"Loading {tsv_path.name} (eager: all rows into memory)")
            examples: list[tuple[int, int, list[int], list[float], int]] = []
            with tsv_path.open("r", encoding="utf-8") as f:
                next(f)  # skip header
                for chunk in tqdm(
                    _iter_line_chunks(f),
                    desc=f"Loading {tsv_path.name}",
                ):
                    examples.extend(_parse_tsv_chunk(chunk, stoi, n_sequences))

            if max_examples is not None and len(examples) > max_examples:
                random.shuffle(examples)
                examples = examples[:max_examples]

            self.examples = examples
            self.offsets: list[int] | None = None
            print(f"Loaded {len(examples):,} rows from {tsv_path.name}")

    def __len__(self) -> int:
        if self.lazy:
            return len(self.offsets)
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.lazy:
            # Open the file handle once per worker process.
            if self._fh is None:
                self._fh = self.tsv_path.open("rb")

            self._fh.seek(self.offsets[idx])
            line = self._fh.readline().decode("utf-8").rstrip("\n")
            seq_idx_s, pos_s, cands_s, scores_s, gold = line.split("\t", maxsplit=4)

            seq_idx = int(seq_idx_s)
            pos = int(pos_s)
            cand_tokens = cands_s.split("\x01")
            cand_ids = [self.stoi.get(t, UNK_ID) for t in cand_tokens]
            kenlm_scores = [float(s) for s in scores_s.split("\x01")]
            try:
                label = cand_tokens.index(gold)
            except ValueError:
                print(
                    f"WARNING: gold token {gold!r} not found in candidates at "
                    f"seq_idx={seq_idx_s}, pos={pos_s}. Marking this row as missing."
                )
                label = MISSING_LABEL
        else:
            seq_idx, pos, cand_ids, kenlm_scores, label = self.examples[idx]

        seq = self.sequences[seq_idx]
        ctx_start = max(0, pos - self.max_context_len)
        context = seq[ctx_start:pos]  # 1-D LongTensor, length 1..max_context_len
        return (
            context,
            torch.tensor(cand_ids, dtype=torch.long),
            torch.tensor(kenlm_scores, dtype=torch.float),
            label,
        )


def collate_precomputed(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate precomputed-candidate examples into padded tensors.

    Simpler than collate_reranker — no negative sampling needed, candidates and
    their KenLM scores are already fixed. Just right-pad contexts and stack tensors.

    Returns:
        context_ids:   [B, T_max] right-padded context token IDs.
        candidate_ids: [B, M] precomputed candidate token IDs.
        kenlm_scores:  [B, M] KenLM log10 probs for each candidate.
        labels:        [B] index of the gold token within each candidate set.
    """
    contexts, candidate_ids_list, kenlm_scores_list, labels = zip(*batch)
    context_ids = torch.nn.utils.rnn.pad_sequence(
        contexts, batch_first=True, padding_value=pad_id
    )  # [B, T_max]
    return (
        context_ids,
        torch.stack(
            candidate_ids_list
        ),  # [B, M] — each element is already a [M] tensor
        torch.stack(kenlm_scores_list),  # [B, M]
        torch.tensor(labels, dtype=torch.long),
    )
