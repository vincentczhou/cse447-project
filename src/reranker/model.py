"""Reranker model and inference utilities (Lightning-free).

Kept Lightning-free so predict.py can import this at inference time without
pulling in training-only dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RerankerConfig


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

        # Fixed KenLM fusion weight (not learned). When precomputed candidates
        # are used, final logits = neural_logits + alpha * kenlm_log10_prob.
        # Set via config.yaml (architecture.alpha). Default 0 = pure neural.
        self.alpha = cfg.alpha

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
            context_ids: [B, T] LongTensor of token IDs (right-padded with pad_id).

        Returns:
            ctx_vec: [B, D] context vector (hidden state at last non-pad position).
        """
        B, T = context_ids.shape
        device = context_ids.device

        # Token + positional embeddings.
        # With right-padding, real tokens start at position 0 and get natural
        # positional encodings regardless of batch max length.
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
        pad_mask = torch.zeros(B, T, device=device, dtype=x.dtype).masked_fill(
            context_ids == self.cfg.pad_id, float("-inf")
        )  # [B, T]

        # Run through Transformer
        x = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True
        )
        x = self.final_norm(x)  # [B, T, D]

        # With right-padding, the last real token is at index (length - 1) per example.
        lengths = (context_ids != self.cfg.pad_id).sum(dim=1)  # [B]
        ctx_vec = x[torch.arange(B, device=device), lengths - 1, :]  # [B, D]
        return ctx_vec

    def score_candidates(
        self,
        context_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        kenlm_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate character given a context prefix.

        Args:
            context_ids:   [B, T] padded context prefix token IDs.
            candidate_ids: [B, M] candidate character token IDs.
            kenlm_scores:  [B, M] KenLM log10 probs. Blended as:
                               logits += alpha * kenlm_scores
                           where alpha is a fixed scalar from config (default 0 = pure neural).
                           Pass zeros when KenLM scores are unavailable (e.g.
                           random-negative training) — alpha has no effect.

        Returns:
            log_probs: [B, M] log-probabilities (higher = more likely). Use F.nll_loss, not F.cross_entropy.
        """
        ctx_vec = self.encode_context(context_ids)  # [B, D]
        cand_emb = self.token_emb(candidate_ids)  # [B, M, D]

        # Dot-product scoring: logit_m = ctx_vec · cand_emb_m
        logits = torch.einsum("bd,bmd->bm", ctx_vec, cand_emb)  # [B, M]

        # Mixture model: blend transformer and KenLM distributions in probability
        # space so they are on the same scale before combining.
        # With alpha=0 this reduces exactly to F.nll_loss(log_softmax(logits/temp))
        # which equals F.cross_entropy(logits/temp) — no behaviour change at alpha=0.
        p_model = F.softmax(logits / self.log_temperature.exp(), dim=-1)  # [B, M]
        p_kenlm = F.softmax(kenlm_scores, dim=-1)  # [B, M]
        blended = (1.0 - self.alpha) * p_model + self.alpha * p_kenlm  # [B, M]

        # Return log-probabilities so callers use F.nll_loss (= cross entropy).
        return blended.log()


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
    # torch.compile wraps the model in a _orig_mod attribute; strip this prefix from state_dict keys so loading works regardless of compilation.
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    stoi = {tok: i for i, tok in enumerate(tokens)}
    return model, tokens, stoi
