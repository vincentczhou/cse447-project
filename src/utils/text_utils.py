"""Shared text preprocessing utilities for KenLM character-level model."""

import re
import unicodedata

SPACE_TOKEN = "<sp>"
_ws_re = re.compile(r"\s+")


def normalize_text(text: str, preserve_trailing_space: bool = False) -> str:
    """Normalize text: fix escaped whitespace, NFC normalize, lowercase, collapse whitespace.

    By default, leading and trailing whitespace are stripped. When
    ``preserve_trailing_space`` is True, one meaningful trailing space is
    retained after whitespace collapsing. This is useful for distillation
    examples where the predicted next character may itself be a space.
    """
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    has_trailing_space = preserve_trailing_space and bool(text) and text[-1].isspace()
    text = _ws_re.sub(" ", text)
    if has_trailing_space:
        return text.lstrip().rstrip() + " "
    return text.strip()


def char_tokenize(text: str) -> str:
    """Convert normalized text to space-separated character tokens."""
    chars = [SPACE_TOKEN if ch == " " else ch for ch in text]
    return " ".join(chars)


def input_to_tokens(text: str) -> list[str]:
    """Convert raw input text to a list of character tokens."""
    normalized = normalize_text(text)
    return [SPACE_TOKEN if ch == " " else ch for ch in normalized]
