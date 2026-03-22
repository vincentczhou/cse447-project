"""Lightning module wrapping the Reranker for training.

Contains RerankerLightningModule (train/val steps, optimizer config) and
load_weights_for_finetuning (loads .pt inference checkpoint weights only).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import lightning as L

from .config import MISSING_LABEL, RerankerConfig, TrainConfig
from .model import Reranker


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
        self.model = Reranker(self.model_cfg)
        self.tokens = tokens
        # register_buffer saves unigram_probs with the checkpoint without
        # treating it as a trainable parameter.
        self.register_buffer("unigram_probs", unigram_probs)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip _orig_mod. prefix from state_dict keys so checkpoints saved
        with torch.compile can be loaded into an uncompiled model (and vice versa)."""
        state = checkpoint.get("state_dict", {})
        fixed = {}
        needs_fix = False
        has_orig = any("_orig_mod." in k for k in state)
        model_compiled = (
            isinstance(self.model, torch._dynamo.eval_frame.OptimizedModule)
            if hasattr(torch, "_dynamo")
            else False
        )
        if has_orig and not model_compiled:
            needs_fix = True
            for k, v in state.items():
                fixed[k.replace("._orig_mod", "")] = v
        elif not has_orig and model_compiled:
            needs_fix = True
            for k, v in state.items():
                if k.startswith("model."):
                    fixed["model._orig_mod." + k[len("model.") :]] = v
                else:
                    fixed[k] = v
        if needs_fix:
            checkpoint["state_dict"] = fixed

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute cross-entropy loss over candidates.

        score_candidates() returns log-probabilities (from the mixture model's
        blended softmax), so F.nll_loss on those log-probs is equivalent to
        cross-entropy loss. Rows where the gold token is missing from the
        candidate set are skipped via ignore_index.
        """
        context_ids, candidate_ids, kenlm_scores, labels = batch
        logits = self.model.score_candidates(context_ids, candidate_ids, kenlm_scores)
        valid_mask = labels != MISSING_LABEL
        if valid_mask.any():
            loss = F.nll_loss(logits, labels, ignore_index=MISSING_LABEL)
        else:
            # If KenLM missed every gold in this batch, there is no supervised
            # signal to learn from. Return a scalar zero on the correct device.
            loss = logits.new_zeros(())
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
        """Log validation loss, top-1/top-3 accuracy, and gold-in-candidates rate."""
        context_ids, candidate_ids, kenlm_scores, labels = batch
        logits = self.model.score_candidates(context_ids, candidate_ids, kenlm_scores)
        valid_mask = labels != MISSING_LABEL
        if valid_mask.any():
            loss = F.nll_loss(logits, labels, ignore_index=MISSING_LABEL)
        else:
            # All rows in this batch are candidate-recall misses, so validation
            # loss is defined as zero rather than forcing an invalid NLL call.
            loss = logits.new_zeros(())
        top1 = ((logits.argmax(dim=1) == labels) & valid_mask).float().mean()
        k = min(3, logits.size(1))
        top3 = (
            (
                (logits.topk(k, dim=1).indices == labels.unsqueeze(1))
                & valid_mask.unsqueeze(1)
            )
            .any(dim=1)
            .float()
            .mean()
        )
        gold_in_candidates = valid_mask.float().mean()
        self.log_dict(
            {
                "valid/loss": loss,
                "valid/top1": top1,
                "valid/top3": top3,
                "valid/gold_in_candidates": gold_in_candidates,
            },
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


def load_weights_for_finetuning(
    lit_model: "RerankerLightningModule", path: str
) -> None:
    """Load model weights from an inference checkpoint (.pt) into a RerankerLightningModule.

    Only loads model parameters — optimizer and scheduler state are left fresh
    so that new hyperparameters (lr, warmup, etc.) take effect from step 0.
    Weights are loaded to CPU first; Lightning moves them to GPU during trainer.fit().
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix if present (from torch.compile checkpoints)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    lit_model.model.load_state_dict(state_dict)
    print(f"Loaded weights from {path} (fresh optimizer/scheduler)")
