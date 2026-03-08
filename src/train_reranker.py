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

import lightning as L
import torch
from dotenv import load_dotenv
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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
        cpu=tr.get("cpu", train_cfg.cpu),
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
        wandb_run_name=wandb_cfg.get("run_name", train_cfg.wandb_run_name),
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
            if (
                len(ids) >= 2
            ):  # need >=1 context token + 1 target token; single-token seqs yield no training examples
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

        # Token + positional embeddings.
        # We use left-padding, so the last real token is always at index T-1,
        # making context vector extraction simple (no variable-length indexing).
        # Tradeoff: real tokens get absolute positions offset by padding length
        # rather than starting from 0, but training and evaluation use the same
        # convention so the model learns to compensate.
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

        # With left-padding the last real token is always at position T-1.
        ctx_vec = x[:, -1, :]  # [B, D]
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
# Lightning module
# ---------------------------------------------------------------------------


class RerankerLightningModule(L.LightningModule):
    """Wraps Reranker as a LightningModule.

    Lightning handles AMP, gradient accumulation, gradient clipping,
    checkpointing, early stopping, and wandb logging — all via Trainer flags
    and callbacks. This class only needs to define the forward logic and
    optimizer configuration.
    """

    def __init__(
        self,
        model_cfg_dict: dict,
        train_cfg_dict: dict,
        tokens: list[str],
        unigram_probs: torch.Tensor,
    ):
        super().__init__()
        # save_hyperparameters embeds model_cfg_dict, train_cfg_dict, and tokens
        # into the Lightning checkpoint so load_from_checkpoint works with no args.
        self.save_hyperparameters(ignore=["unigram_probs", "tokens"])
        self.model_cfg = RerankerConfig(**model_cfg_dict)
        self.train_cfg = TrainConfig(**train_cfg_dict)
        self.model = torch.compile(Reranker(self.model_cfg))
        self.tokens = tokens
        # register_buffer saves unigram_probs with the checkpoint without
        # treating it as a trainable parameter.
        self.register_buffer("unigram_probs", unigram_probs)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        context_ids, candidate_ids, labels = batch
        logits = self.model.score_candidates(context_ids, candidate_ids)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        context_ids, candidate_ids, labels = batch
        logits = self.model.score_candidates(context_ids, candidate_ids)
        loss = F.cross_entropy(logits, labels)
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        k = min(3, logits.size(1))
        top3 = (
            (logits.topk(k, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
        )
        self.log_dict(
            {"valid/loss": loss, "valid/top1": top1, "valid/top3": top3},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        # Compute total steps for scheduler (trainer.estimated_stepping_batches
        # accounts for grad accumulation and epochs automatically).
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.train_cfg.warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        min_lr = self.train_cfg.min_lr_ratio * self.train_cfg.lr

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        if self.train_cfg.lr_scheduler == "cosine":
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=min_lr
            )
        else:  # linear
            decay = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.train_cfg.min_lr_ratio,
                total_iters=decay_steps,
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Inference loading  (Lightning-free — no Lightning needed at inference time)
# ---------------------------------------------------------------------------


def load_for_inference(
    path: Path, device: torch.device
) -> tuple[Reranker, list[str], dict[str, int]]:
    """Load a plain inference checkpoint (saved by main() after training).

    The inference checkpoint is a plain PyTorch file — no Lightning required.
    It contains: model_state_dict, tokens, config (as dict).

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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load config from config.yaml and run training. No CLI arguments."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model_cfg, train_cfg = _load_config_from_yaml()

    # ---- Seed ----
    L.seed_everything(train_cfg.seed)

    # ---- Vocab & sequences ----
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
    use_gpu = not train_cfg.cpu and torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_gpu,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_gpu,
    )

    # ---- Lightning module ----
    lit_model = RerankerLightningModule(
        dataclasses.asdict(model_cfg),
        dataclasses.asdict(train_cfg),
        tokens,
        unigram_probs,
    )
    param_count = sum(p.numel() for p in lit_model.model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ---- Callbacks ----
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename="best_reranker",
            monitor="valid/top3",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="valid/top3",
            patience=train_cfg.patience,
            mode="max",
        ),
    ]

    # ---- Logger ----
    if train_cfg.wandb_project:
        try:
            logger = WandbLogger(
                project=train_cfg.wandb_project,
                entity=train_cfg.wandb_entity,
                name=train_cfg.wandb_run_name,
            )
        except Exception:
            print("wandb unavailable, disabling logger")
            logger = True
    else:
        logger = True  # Lightning's default CSV logger

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=train_cfg.epochs,
        precision="16-mixed" if (train_cfg.mixed_precision and use_gpu) else 32,
        accumulate_grad_batches=train_cfg.gradient_accumulation_steps,
        gradient_clip_val=train_cfg.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=train_cfg.log_every,
        default_root_dir=str(out_dir),
        accelerator="gpu" if use_gpu else "cpu",
        devices="auto",
        strategy="ddp",
        limit_val_batches=train_cfg.max_eval_batches or 1.0,
    )

    trainer.fit(lit_model, train_loader, valid_loader, ckpt_path=train_cfg.resume_from)

    # ---- Export plain inference checkpoint (no Lightning needed at runtime) ----
    inference_path = out_dir / train_cfg.checkpoint_name
    torch.save(
        {
            "model_state_dict": lit_model.model.state_dict(),
            "tokens": tokens,
            "config": dataclasses.asdict(model_cfg),
        },
        inference_path,
    )
    print(f"Inference checkpoint saved to {inference_path}")


if __name__ == "__main__":
    main()
