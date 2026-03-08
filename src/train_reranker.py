#!/usr/bin/env python
"""
Train a character-level neural reranker for next-character prediction.

All configuration is loaded from config.yaml (reranker section). No CLI arguments.

Expected input files (from src/data/preprocess.py):
- data/.../train.txt  (space-separated character tokens, one document per line)
- data/.../valid.txt  (space-separated character tokens)
- data/.../vocab.json ({token: count} sorted by frequency descending)
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RerankerConfig:
    """Model architecture config. Saved with checkpoint for inference loading."""

    vocab_size: int = 0  # filled after loading vocab
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_context_len: int = 256
    pad_id: int = PAD_ID
    norm_first: bool = True
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters. NOT saved with checkpoint."""

    # Paths
    train_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/train.txt"
    valid_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/valid.txt"
    vocab_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/vocab.json"
    out_dir: str = "work"
    checkpoint_name: str = "reranker.pt"
    resume_from: str | None = None

    # Training
    epochs: int = 10
    batch_size: int = 64
    eval_batch_size: int = 128
    candidate_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True

    # LR scheduling
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1

    # Early stopping / best model
    patience: int = 3
    metric: str = "valid_top3"

    # Data limits
    max_train_lines: int | None = None
    max_valid_lines: int | None = None
    max_train_examples: int | None = None
    max_valid_examples: int | None = 50_000
    max_eval_batches: int | None = 200

    # Wandb
    wandb_project: str | None = "cse447"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    # Misc
    seed: int = 42
    log_every: int = 100
    num_workers: int = 2
    cpu: bool = False


def _load_config_from_yaml() -> tuple[RerankerConfig, TrainConfig]:
    """Load reranker config from config.yaml, returning defaults if missing."""
    model_cfg = RerankerConfig()
    train_cfg = TrainConfig()

    if not _CONFIG_PATH.exists():
        return model_cfg, train_cfg

    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rcfg = raw.get("reranker", {})
    if not rcfg:
        return model_cfg, train_cfg

    # Architecture -> RerankerConfig
    arch = rcfg.get("architecture", {})
    if arch:
        model_cfg = RerankerConfig(
            d_model=arch.get("d_model", model_cfg.d_model),
            nhead=arch.get("nhead", model_cfg.nhead),
            num_layers=arch.get("num_layers", model_cfg.num_layers),
            ff_mult=arch.get("ff_mult", model_cfg.ff_mult),
            dropout=arch.get("dropout", model_cfg.dropout),
            max_context_len=arch.get("max_context_len", model_cfg.max_context_len),
            norm_first=arch.get("norm_first", model_cfg.norm_first),
            temperature=arch.get("temperature", model_cfg.temperature),
        )

    # Training + early_stopping + data + data_limits + output + logging -> TrainConfig
    tr = rcfg.get("training", {})
    es = rcfg.get("early_stopping", {})
    data = rcfg.get("data", {})
    dl = rcfg.get("data_limits", {})
    out = rcfg.get("output", {})
    log = rcfg.get("logging", {})
    wandb_cfg = raw.get("wandb", {})

    train_cfg = TrainConfig(
        train_path=data.get("train_path", train_cfg.train_path),
        valid_path=data.get("valid_path", train_cfg.valid_path),
        vocab_path=data.get("vocab_path", train_cfg.vocab_path),
        out_dir=out.get("out_dir", train_cfg.out_dir),
        checkpoint_name=out.get("checkpoint_name", train_cfg.checkpoint_name),
        resume_from=out.get("resume_from", train_cfg.resume_from),
        epochs=tr.get("epochs", train_cfg.epochs),
        batch_size=tr.get("batch_size", train_cfg.batch_size),
        eval_batch_size=tr.get("eval_batch_size", train_cfg.eval_batch_size),
        candidate_size=tr.get("candidate_size", train_cfg.candidate_size),
        lr=tr.get("lr", train_cfg.lr),
        weight_decay=tr.get("weight_decay", train_cfg.weight_decay),
        grad_clip=tr.get("grad_clip", train_cfg.grad_clip),
        warmup_steps=tr.get("warmup_steps", train_cfg.warmup_steps),
        gradient_accumulation_steps=tr.get(
            "gradient_accumulation_steps", train_cfg.gradient_accumulation_steps
        ),
        mixed_precision=tr.get("mixed_precision", train_cfg.mixed_precision),
        lr_scheduler=tr.get("lr_scheduler", train_cfg.lr_scheduler),
        min_lr_ratio=tr.get("min_lr_ratio", train_cfg.min_lr_ratio),
        seed=tr.get("seed", train_cfg.seed),
        patience=es.get("patience", train_cfg.patience),
        metric=es.get("metric", train_cfg.metric),
        max_train_lines=dl.get("max_train_lines", train_cfg.max_train_lines),
        max_valid_lines=dl.get("max_valid_lines", train_cfg.max_valid_lines),
        max_train_examples=dl.get("max_train_examples", train_cfg.max_train_examples),
        max_valid_examples=dl.get("max_valid_examples", train_cfg.max_valid_examples),
        max_eval_batches=dl.get("max_eval_batches", train_cfg.max_eval_batches),
        wandb_project=wandb_cfg.get("project", train_cfg.wandb_project)
        if wandb_cfg.get("enabled", True)
        else None,
        wandb_entity=wandb_cfg.get("entity", train_cfg.wandb_entity),
        log_every=log.get("log_every", train_cfg.log_every),
        num_workers=log.get("num_workers", train_cfg.num_workers),
    )

    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Vocabulary & sequence loading
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: Path) -> tuple[list[str], torch.Tensor]:
    """Load vocab.json, prepend <pad> and <unk>, return tokens and unigram probs.

    Returns:
        tokens: list of token strings indexed by ID.
        unigram_probs: [V] tensor of normalized probabilities (pad=0).
    """
    # TODO(kenlm-integration): Currently returns unigram frequency probs for
    # random negative sampling.  Once KenLM is integrated, replace this with
    # a loader that reads precomputed per-position KenLM top-K candidate sets
    # so the reranker trains on realistic "hard" negatives instead of random
    # frequency-weighted ones.  See also: collate_reranker().
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
    """Load tokenized text file into a list of 1-D LongTensors."""
    sequences: list[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = [stoi.get(tok, UNK_ID) for tok in line.split()]
            if len(ids) >= 2:
                sequences.append(torch.tensor(ids, dtype=torch.long))
    return sequences


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Reranker(nn.Module):
    """Causal Transformer reranker for next-character prediction.

    Architecture:
        1. Shared embedding layer maps token IDs -> d_model vectors.
        2. Learned positional embeddings (up to max_context_len positions).
        3. Causal (GPT-style) Transformer encoder processes the context prefix.
        4. Context vector = hidden state at the last non-pad position.
        5. Candidate scores = dot product of context vector with each candidate
           embedding, divided by a learnable temperature.

    The model never produces a full-vocab softmax. It scores only the M
    candidates provided to it (restricted softmax), making it efficient.
    """

    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg

        # --- Embeddings (shared between context and candidates) ---
        self.token_emb = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id
        )
        self.pos_emb = nn.Embedding(cfg.max_context_len, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # --- Causal Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,  # nested tensors conflict with causal masks
        )

        # Final layer norm (needed when using pre-norm, since the last encoder
        # layer's output is un-normalized)
        if cfg.norm_first:
            self.final_norm = nn.LayerNorm(cfg.d_model)
        else:
            self.final_norm = nn.Identity()

        # Learnable temperature for logit scaling
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(cfg.temperature), dtype=torch.float)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights with small normal distribution."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.token_emb.weight[self.cfg.pad_id].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def encode_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Encode a padded context prefix into a single context vector per example.

        Args:
            context_ids: [B, T] LongTensor of token IDs (left-padded with pad_id).

        Returns:
            ctx_vec: [B, D] context vector (hidden state at last non-pad position).
        """
        B, T = context_ids.shape
        device = context_ids.device

        # Token + positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        x = self.token_emb(context_ids) + self.pos_emb(positions)  # [B, T, D]
        x = self.emb_drop(x)

        # Causal mask: upper-triangular -inf so each position only attends to
        # itself and earlier positions.  Float mask (not bool) for SDPA compatibility.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device, dtype=x.dtype
        )  # [T, T]

        # Padding mask: -inf for pad positions, 0.0 for real tokens.
        # Shape [B, T] — PyTorch broadcasts this over attention heads.
        pad_mask = torch.where(
            context_ids == self.cfg.pad_id,
            torch.tensor(float("-inf"), device=device, dtype=x.dtype),
            torch.tensor(0.0, device=device, dtype=x.dtype),
        )  # [B, T]

        # Run through Transformer
        x = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True
        )
        x = self.final_norm(x)  # [B, T, D]

        # Extract the last non-pad position's hidden state for each example.
        # lengths[b] = number of non-pad tokens in example b.
        lengths = (context_ids != self.cfg.pad_id).sum(dim=1)  # [B]
        last_idx = (lengths - 1).clamp(min=0)  # [B]
        ctx_vec = x[torch.arange(B, device=device), last_idx]  # [B, D]
        return ctx_vec

    def score_candidates(
        self, context_ids: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score each candidate character given a context prefix.

        Args:
            context_ids:  [B, T] padded context prefix token IDs.
            candidate_ids: [B, M] candidate character token IDs.

        Returns:
            logits: [B, M] unnormalized scores (higher = more likely).
        """
        ctx_vec = self.encode_context(context_ids)  # [B, D]
        cand_emb = self.token_emb(candidate_ids)  # [B, M, D]

        # Dot-product scoring: logit_m = ctx_vec · cand_emb_m
        logits = torch.einsum("bd,bmd->bm", ctx_vec, cand_emb)  # [B, M]

        # Temperature scaling
        temperature = self.log_temperature.exp()
        logits = logits / temperature

        return logits


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
        self.index: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(sequences):
            for pos in range(1, len(seq)):
                self.index.append((seq_idx, pos))

        # Optionally cap the number of examples (for quick iteration).
        if max_examples is not None and len(self.index) > max_examples:
            random.shuffle(self.index)
            self.index = self.index[:max_examples]

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a batch of (context, target) pairs into padded tensors with
    vectorized negative sampling.

    For each example, builds a candidate set of size M:
        - 1 gold target (placed at a random position)
        - M-1 negative samples drawn from unigram_probs

    Args:
        batch: list of (context_ids: [T_i], target_id: int) from RerankerDataset.
        candidate_size: M, total number of candidates per example.
        unigram_probs: [V] tensor of sampling probabilities for negatives.
        pad_id: token ID used for left-padding contexts.

    Returns:
        context_ids:   [B, T_max] left-padded context token IDs.
        candidate_ids: [B, M] candidate token IDs (gold is hidden among negatives).
        labels:        [B] index of the gold target within each candidate set.
    """
    # TODO(kenlm-integration): Replace unigram negative sampling with
    # precomputed KenLM top-K candidate sets for realistic hard negatives.
    # When that happens, this function would read candidates from the batch
    # instead of sampling them here.

    contexts, targets = zip(*batch)
    B = len(contexts)
    M = candidate_size

    # --- Left-pad contexts to the longest in the batch ---
    max_len = max(c.size(0) for c in contexts)
    context_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, c in enumerate(contexts):
        context_ids[i, max_len - c.size(0) :] = c

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

    return context_ids, candidate_ids, labels


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create an LR scheduler with linear warmup then decay.

    Args:
        name: "cosine" or "linear".
        optimizer: the optimizer whose LR to schedule.
        warmup_steps: number of steps for linear warmup (0 -> base LR).
        total_steps: total training steps (warmup + decay).
        min_lr_ratio: final LR = base_lr * min_lr_ratio.

    Returns:
        A LambdaLR scheduler (call .step() once per optimizer step).
    """

    def lr_lambda(current_step: int) -> float:
        # Phase 1: linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Phase 2: decay from 1.0 down to min_lr_ratio
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)

        if name == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif name == "linear":
            decay = 1.0 - progress
        else:
            raise ValueError(f"Unknown scheduler: {name!r}. Use 'cosine' or 'linear'.")

        # Scale decay so it goes from 1.0 -> min_lr_ratio (not 1.0 -> 0.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: Reranker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    global_step: int,
    model_config: RerankerConfig,
    tokens: list[str],
) -> None:
    """Save a full checkpoint (training resume + inference loading).

    The checkpoint contains two groups of data:
        Inference-ready: model_state_dict, tokens, config (as dict)
        Resume-only:     optimizer, scheduler, scaler, epoch, best_metric, global_step
    """
    ckpt = {
        # Inference-ready fields (used by myprogram.py later)
        "model_state_dict": model.state_dict(),
        "tokens": tokens,
        "config": dataclasses.asdict(model_config),
        # Training resume fields
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "global_step": global_step,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: Reranker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, float, int]:
    """Load a checkpoint for training resumption.

    Args:
        path: path to the checkpoint file.
        model, optimizer, scheduler, scaler: objects to load state into.
        device: device to map tensors to.

    Returns:
        (next_epoch, best_metric, global_step) so the training loop can
        continue from where it left off.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["best_metric"], ckpt["global_step"]


def load_for_inference(
    path: Path, device: torch.device
) -> tuple[Reranker, list[str], dict[str, int]]:
    """Load a checkpoint for inference (no optimizer/scheduler needed).

    Reconstructs the model from the saved config, loads weights, and
    returns everything needed to call model.score_candidates().

    Returns:
        model: Reranker in eval mode on the given device.
        tokens: list of token strings indexed by ID.
        stoi: {token_string: token_id} mapping.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = RerankerConfig(**ckpt["config"])
    tokens = ckpt["tokens"]

    model = Reranker(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    stoi = {tok: i for i, tok in enumerate(tokens)}
    return model, tokens, stoi


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: Reranker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    autocast_ctx,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate the model on a validation DataLoader.

    Args:
        model: the Reranker (will be set to eval mode, then restored).
        loader: DataLoader yielding (context_ids, candidate_ids, labels).
        device: device tensors should be on.
        autocast_ctx: torch.amp.autocast context (or nullcontext).
        max_batches: cap on number of batches to evaluate (None = all).

    Returns:
        Dict with keys: "loss", "top1_acc", "top3_acc".
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_examples = 0

    for batch_idx, (context_ids, candidate_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        context_ids = context_ids.to(device)
        candidate_ids = candidate_ids.to(device)
        labels = labels.to(device)

        with autocast_ctx:
            logits = model.score_candidates(context_ids, candidate_ids)  # [B, M]
            loss = F.cross_entropy(logits, labels)

        B = labels.size(0)
        total_loss += loss.item() * B
        total_examples += B

        # Top-1: is the highest-scoring candidate the gold?
        top1_preds = logits.argmax(dim=1)  # [B]
        total_top1 += (top1_preds == labels).sum().item()

        # Top-3: is gold among the 3 highest-scoring candidates?
        top3_preds = logits.topk(min(3, logits.size(1)), dim=1).indices  # [B, 3]
        total_top3 += (top3_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

    if was_training:
        model.train()

    if total_examples == 0:
        return {"loss": 0.0, "top1_acc": 0.0, "top3_acc": 0.0}

    return {
        "loss": total_loss / total_examples,
        "top1_acc": total_top1 / total_examples,
        "top3_acc": total_top3 / total_examples,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model_cfg: RerankerConfig, train_cfg: TrainConfig) -> None:
    """Full training loop: data -> model -> train -> evaluate -> checkpoint."""

    # ---- Seed & device ----
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    if train_cfg.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device: {device}")

    # ---- Load vocab & data ----
    tokens, unigram_probs = load_vocab(Path(train_cfg.vocab_path))
    model_cfg.vocab_size = len(tokens)
    stoi = {tok: i for i, tok in enumerate(tokens)}
    print(f"Vocab: {len(tokens)} tokens")

    train_seqs = load_sequences(
        Path(train_cfg.train_path), stoi, train_cfg.max_train_lines
    )
    valid_seqs = load_sequences(
        Path(train_cfg.valid_path), stoi, train_cfg.max_valid_lines
    )
    print(f"Train: {len(train_seqs)} sequences, Valid: {len(valid_seqs)} sequences")

    # ---- Datasets & DataLoaders ----
    train_ds = RerankerDataset(
        train_seqs, model_cfg.max_context_len, train_cfg.max_train_examples
    )
    valid_ds = RerankerDataset(
        valid_seqs, model_cfg.max_context_len, train_cfg.max_valid_examples
    )
    print(f"Train examples: {len(train_ds)}, Valid examples: {len(valid_ds)}")

    collate_fn = partial(
        collate_reranker,
        candidate_size=train_cfg.candidate_size,
        unigram_probs=unigram_probs,
        pad_id=model_cfg.pad_id,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model, optimizer, scheduler, scaler ----
    model = Reranker(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    steps_per_epoch = len(train_loader) // train_cfg.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_cfg.epochs
    scheduler = get_scheduler(
        train_cfg.lr_scheduler,
        optimizer,
        train_cfg.warmup_steps,
        total_steps,
        train_cfg.min_lr_ratio,
    )

    use_amp = train_cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_amp
        else torch.amp.autocast("cpu", enabled=False)
    )

    # ---- Output directory ----
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / train_cfg.checkpoint_name
    best_ckpt_path = out_dir / f"best_{train_cfg.checkpoint_name}"

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_metric = -float("inf") if "top" in train_cfg.metric else float("inf")
    global_step = 0

    if train_cfg.resume_from is not None:
        resume_path = Path(train_cfg.resume_from)
        print(f"Resuming from {resume_path}")
        start_epoch, best_metric, global_step = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        print(
            f"  Resumed at epoch {start_epoch}, step {global_step}, best_metric={best_metric:.4f}"
        )

    # ---- Wandb ----
    if train_cfg.wandb_project:
        try:
            import wandb

            wandb.init(
                project=train_cfg.wandb_project,
                entity=train_cfg.wandb_entity,
                name=train_cfg.wandb_run_name,
                config={
                    "model": dataclasses.asdict(model_cfg),
                    "training": dataclasses.asdict(train_cfg),
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            train_cfg.wandb_project = None

    higher_is_better = "top" in train_cfg.metric
    patience_counter = 0

    # ---- Training loop ----
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, (context_ids, candidate_ids, labels) in enumerate(train_loader):
            context_ids = context_ids.to(device)
            candidate_ids = candidate_ids.to(device)
            labels = labels.to(device)

            with autocast_ctx:
                logits = model.score_candidates(context_ids, candidate_ids)
                loss = F.cross_entropy(logits, labels)
                loss = loss / train_cfg.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * train_cfg.gradient_accumulation_steps
            epoch_batches += 1

            # Optimizer step every gradient_accumulation_steps
            is_accum_step = (batch_idx + 1) % train_cfg.gradient_accumulation_steps == 0
            is_last_batch = batch_idx == len(train_loader) - 1
            if is_accum_step or is_last_batch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg.grad_clip
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic logging
                if global_step % train_cfg.log_every == 0:
                    avg_loss = epoch_loss / epoch_batches
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  epoch {epoch + 1}/{train_cfg.epochs}  "
                        f"step {global_step}  "
                        f"loss {avg_loss:.4f}  "
                        f"lr {lr:.2e}"
                    )
                    if train_cfg.wandb_project:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )

        # ---- End-of-epoch evaluation ----
        avg_train_loss = epoch_loss / max(1, epoch_batches)
        val_metrics = evaluate(
            model, valid_loader, device, autocast_ctx, train_cfg.max_eval_batches
        )

        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"valid_loss={val_metrics['loss']:.4f}  "
            f"valid_top1={val_metrics['top1_acc']:.4f}  "
            f"valid_top3={val_metrics['top3_acc']:.4f}"
        )

        if train_cfg.wandb_project:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "valid/loss": val_metrics["loss"],
                    "valid/top1_acc": val_metrics["top1_acc"],
                    "valid/top3_acc": val_metrics["top3_acc"],
                },
                step=global_step,
            )

        # ---- Best model tracking & early stopping ----
        if train_cfg.metric == "valid_top3":
            current_metric = val_metrics["top3_acc"]
        elif train_cfg.metric == "valid_loss":
            current_metric = val_metrics["loss"]
        else:
            current_metric = val_metrics.get(train_cfg.metric, val_metrics["top3_acc"])

        improved = (
            current_metric > best_metric
            if higher_is_better
            else current_metric < best_metric
        )

        if improved:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                global_step,
                model_cfg,
                tokens,
            )
            print(
                f"  -> New best {train_cfg.metric}={best_metric:.4f}, saved to {best_ckpt_path}"
            )
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{train_cfg.patience})")

        # Save latest checkpoint every epoch
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            global_step,
            model_cfg,
            tokens,
        )

        if patience_counter >= train_cfg.patience:
            print(
                f"Early stopping after {epoch + 1} epochs (no improvement for {train_cfg.patience} epochs)"
            )
            break

    # ---- Cleanup ----
    if train_cfg.wandb_project:
        import wandb

        wandb.finish()

    print(f"Training complete. Best {train_cfg.metric}={best_metric:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Latest checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load config from config.yaml and run training. No CLI arguments."""
    model_cfg, train_cfg = _load_config_from_yaml()
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
