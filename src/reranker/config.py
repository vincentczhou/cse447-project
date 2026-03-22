"""Configuration dataclasses and YAML loading for the reranker.

Defines RerankerConfig (model architecture, saved with checkpoints) and
TrainConfig (training hyperparameters, not saved). Both are populated from
config.yaml via _load_config_from_yaml().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MISSING_LABEL = -1

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

PAD_ID = 0
UNK_ID = 1

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
    alpha: float = 0.0


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters. NOT saved with checkpoint."""

    # Paths
    train_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/train.txt"
    valid_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/valid.txt"
    vocab_path: str = "data/madlad_multilang_clean_15k_optionB_kenlm/vocab.json"
    # Precomputed KenLM candidate TSVs (from precompute_kenlm_candidates.py).
    # When both are set and exist, hard KenLM negatives are used instead of random
    # negatives. candidate_size is ignored — K comes from the TSV's --k flag.
    candidates_train_path: str | None = None
    candidates_valid_path: str | None = None
    out_dir: str = "work"
    checkpoint_name: str = "reranker.pt"
    resume_from: str | None = None
    init_weights_from: str | None = (
        None  # Load weights only (no optimizer/scheduler); uses .pt inference checkpoint
    )

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
    metric: str = "valid/top3"
    metric_mode: str = "max"  # "max" for top3, "min" for loss

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

    # Checkpointing
    every_n_train_steps: int | None = None
    save_on_train_epoch_end: bool | None = None

    # Validation frequency (int = every N optimizer steps, 1.0 = every epoch)
    val_check_interval: int | float | None = None

    # Misc
    seed: int = 42
    log_every: int = 100
    num_workers: int = 2
    pin_memory: bool = True
    cpu: bool = False
    lazy_load_candidates: bool = (
        True  # False = load all TSV rows into memory (faster __getitem__)
    )


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
            alpha=arch.get("alpha", model_cfg.alpha),
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
        init_weights_from=out.get("init_weights_from", train_cfg.init_weights_from),
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
        metric_mode=es.get("metric_mode", train_cfg.metric_mode),
        max_train_lines=dl.get("max_train_lines", train_cfg.max_train_lines),
        max_valid_lines=dl.get("max_valid_lines", train_cfg.max_valid_lines),
        max_train_examples=dl.get("max_train_examples", train_cfg.max_train_examples),
        max_valid_examples=dl.get("max_valid_examples", train_cfg.max_valid_examples),
        max_eval_batches=dl.get("max_eval_batches", train_cfg.max_eval_batches),
        candidates_train_path=data.get(
            "candidates_train_path", train_cfg.candidates_train_path
        ),
        candidates_valid_path=data.get(
            "candidates_valid_path", train_cfg.candidates_valid_path
        ),
        wandb_project=wandb_cfg.get("project", train_cfg.wandb_project)
        if wandb_cfg.get("enabled", True)
        else None,
        wandb_entity=wandb_cfg.get("entity", train_cfg.wandb_entity),
        wandb_run_name=wandb_cfg.get("run_name", train_cfg.wandb_run_name),
        log_every=log.get("log_every", train_cfg.log_every),
        num_workers=log.get("num_workers", train_cfg.num_workers),
        pin_memory=log.get("pin_memory", train_cfg.pin_memory),
        every_n_train_steps=out.get(
            "every_n_train_steps", train_cfg.every_n_train_steps
        ),
        save_on_train_epoch_end=out.get(
            "save_on_train_epoch_end", train_cfg.save_on_train_epoch_end
        ),
        val_check_interval=log.get("val_check_interval", train_cfg.val_check_interval),
        lazy_load_candidates=data.get(
            "lazy_load_candidates", train_cfg.lazy_load_candidates
        ),
    )

    return model_cfg, train_cfg
