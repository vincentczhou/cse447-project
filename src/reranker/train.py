"""Training entry point for the reranker.

Run with:
    uv run python src/reranker/train.py
    uv run python -m reranker.train   (from repo root with src/ on sys.path)
"""

from __future__ import annotations

import dataclasses
import sys
from functools import partial
from pathlib import Path

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reranker.config import _load_config_from_yaml
from reranker.dataset import (
    PrecomputedRerankerDataset,
    RerankerDataset,
    collate_precomputed,
    collate_reranker,
    load_sequences,
    load_vocab,
)
from reranker.lightning_module import (
    RerankerLightningModule,
    load_weights_for_finetuning,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load config from config.yaml and run training. No CLI arguments."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    model_cfg, train_cfg = _load_config_from_yaml()

    # ---- Seed ----
    L.seed_everything(train_cfg.seed)

    # ---- Vocab & sequences ----
    tokens, unigram_probs = load_vocab(Path(train_cfg.vocab_path))
    model_cfg.vocab_size = len(tokens)
    stoi = {tok: i for i, tok in enumerate(tokens)}
    print(f"Vocab: {len(tokens)} tokens")

    print(f"Loading sequences from {train_cfg.train_path}")
    train_seqs = load_sequences(
        Path(train_cfg.train_path),
        stoi,
        train_cfg.max_train_lines,
    )
    print(f"Loading sequences from {train_cfg.valid_path}")
    valid_seqs = load_sequences(
        Path(train_cfg.valid_path),
        stoi,
        train_cfg.max_valid_lines,
    )
    print(f"Train: {len(train_seqs)} sequences, Valid: {len(valid_seqs)} sequences")

    # ---- Datasets & DataLoaders ----
    cand_train = (
        Path(train_cfg.candidates_train_path)
        if train_cfg.candidates_train_path
        else None
    )
    cand_valid = (
        Path(train_cfg.candidates_valid_path)
        if train_cfg.candidates_valid_path
        else None
    )
    use_precomputed = (
        cand_train is not None
        and cand_train.exists()
        and cand_valid is not None
        and cand_valid.exists()
    )

    if use_precomputed:
        print(
            f"Using precomputed KenLM candidates: {cand_train.name}, {cand_valid.name}"
        )
        print(f"Loading train candidates from {cand_train}")
        train_ds = PrecomputedRerankerDataset(
            cand_train,
            train_seqs,
            stoi,
            model_cfg.max_context_len,
            train_cfg.max_train_examples,
            lazy=train_cfg.lazy_load_candidates,
        )
        print(f"Loading valid candidates from {cand_valid}")
        valid_ds = PrecomputedRerankerDataset(
            cand_valid,
            valid_seqs,
            stoi,
            model_cfg.max_context_len,
            train_cfg.max_valid_examples,
            lazy=train_cfg.lazy_load_candidates,
        )
        collate_fn = partial(collate_precomputed, pad_id=model_cfg.pad_id)
    else:
        print(
            "Using random frequency-weighted negatives (no precomputed candidates found)"
        )
        train_ds = RerankerDataset(
            train_seqs, model_cfg.max_context_len, train_cfg.max_train_examples
        )
        valid_ds = RerankerDataset(
            valid_seqs, model_cfg.max_context_len, train_cfg.max_valid_examples
        )
        collate_fn = partial(
            collate_reranker,
            candidate_size=train_cfg.candidate_size,
            unigram_probs=unigram_probs,
            pad_id=model_cfg.pad_id,
        )
    print(f"Train examples: {len(train_ds)}, Valid examples: {len(valid_ds)}")
    use_gpu = not train_cfg.cpu and torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=train_cfg.pin_memory and use_gpu,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=train_cfg.pin_memory and use_gpu,
    )

    # ---- Lightning module ----
    lit_model = RerankerLightningModule(
        dataclasses.asdict(model_cfg),
        dataclasses.asdict(train_cfg),
        tokens,
        unigram_probs,
    )
    if train_cfg.init_weights_from:
        load_weights_for_finetuning(lit_model, train_cfg.init_weights_from)

    param_count = sum(p.numel() for p in lit_model.model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ---- Callbacks ----
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename="best_reranker",
            monitor=train_cfg.metric,
            mode=train_cfg.metric_mode,
            save_top_k=1,
            every_n_train_steps=train_cfg.every_n_train_steps,
            save_on_train_epoch_end=train_cfg.save_on_train_epoch_end,
        ),
        EarlyStopping(
            monitor=train_cfg.metric,
            patience=train_cfg.patience,
            mode=train_cfg.metric_mode,
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
        strategy="auto",
        limit_val_batches=train_cfg.max_eval_batches or 1.0,
        val_check_interval=train_cfg.val_check_interval,
    )

    try:
        trainer.fit(
            lit_model, train_loader, valid_loader, ckpt_path=train_cfg.resume_from
        )
    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted — saving inference checkpoint...")
    finally:
        pass

    # ---- Export plain inference checkpoint (no Lightning needed at runtime) ----
    inference_path = out_dir / train_cfg.checkpoint_name
    # torch.compile wraps the model in a _orig_mod attribute; strip this prefix from state_dict keys so loading works regardless of compilation.
    state_dict = lit_model.model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    torch.save(
        {
            "model_state_dict": state_dict,
            "tokens": tokens,
            "config": dataclasses.asdict(model_cfg),
        },
        inference_path,
    )
    print(f"Inference checkpoint saved to {inference_path}")


if __name__ == "__main__":
    main()
