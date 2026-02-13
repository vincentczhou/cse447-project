"""Shared text preprocessing utilities for KenLM character-level model."""

import re
import unicodedata

SPACE_TOKEN = "<sp>"
_ws_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text: fix escaped whitespace, NFC normalize, lowercase, collapse whitespace."""
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _ws_re.sub(" ", text).strip()
    return text


def char_tokenize(text: str) -> str:
    """Convert normalized text to space-separated character tokens."""
    chars = [SPACE_TOKEN if ch == " " else ch for ch in text]
    return " ".join(chars)


def input_to_tokens(text: str) -> list[str]:
    """Convert raw input text to a list of character tokens."""
    normalized = normalize_text(text)
    return [SPACE_TOKEN if ch == " " else ch for ch in normalized]
