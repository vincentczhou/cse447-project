"""Convert a Lightning .ckpt checkpoint to a plain inference .pt file.

Usage:
    uv run python scripts/ckpt_to_pt.py work/best_reranker.ckpt work/reranker.pt
"""

import dataclasses
import sys
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from reranker import RerankerLightningModule, load_vocab  # noqa: E402


def convert(ckpt_path: Path, out_path: Path, vocab_path: Path) -> None:
    """Load a Lightning checkpoint and save a plain inference .pt file."""
    print(f"Loading {ckpt_path}")
    tokens, unigram_probs = load_vocab(vocab_path)
    lit_model = RerankerLightningModule.load_from_checkpoint(
        ckpt_path, tokens=tokens, unigram_probs=unigram_probs
    )
    lit_model.eval()

    state_dict = lit_model.model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    torch.save(
        {
            "model_state_dict": state_dict,
            "tokens": tokens,
            "config": dataclasses.asdict(lit_model.model_cfg),
        },
        out_path,
    )
    print(f"Saved inference checkpoint to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Lightning .ckpt to inference .pt"
    )
    parser.add_argument("ckpt", type=Path, help="input .ckpt file")
    parser.add_argument("out", type=Path, help="output .pt file")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="vocab JSON path (defaults to config.yaml model.vocab under work/)",
    )
    args = parser.parse_args()

    if args.vocab is None:
        cfg = yaml.safe_load((_REPO / "config.yaml").read_text())
        args.vocab = _REPO / "work" / cfg["model"]["vocab"]

    convert(args.ckpt, args.out, args.vocab)
