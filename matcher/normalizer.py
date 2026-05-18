"""
normalizer.py — Unicode normalization, diacritic stripping, punctuation cleanup
Handles: French, Spanish, German, Arabic transliteration, hyphens, case
"""

import re
import unicodedata


def normalize(name: str) -> str:
    """
    Full normalization pipeline:
    1. Strip leading/trailing whitespace
    2. Unicode NFC normalization
    3. Diacritic / accent removal (NFD decompose → strip combining chars)
    4. Lowercase
    5. Replace hyphens and underscores with space
    6. Remove punctuation except periods (for initials)
    7. Collapse multiple spaces
    """
    if not name or not isinstance(name, str):
        return ""

    # Strip whitespace
    name = name.strip()

    # Unicode NFC first
    name = unicodedata.normalize("NFC", name)

    # NFD decompose → strip combining diacritical marks
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # Lowercase
    name = name.lower()

    # Replace hyphens, underscores, slashes with space
    name = re.sub(r"[-_/\\]", " ", name)

    # Remove punctuation except periods (keep for initials like J.)
    name = re.sub(r"[^\w\s.]", "", name)

    # Remove standalone periods not used as initials (e.g. "..." → "")
    # Keep: "j." or "a." — single char followed by period
    name = re.sub(r"(?<!\b\w)\.", "", name)

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def tokenize(name: str) -> list[str]:
    """Split normalized name into tokens, filtering empty strings."""
    return [t for t in name.split(" ") if t]


def is_initial(token: str) -> bool:
    """Check if a token is an initial — single char optionally followed by period."""
    token = token.rstrip(".")
    return len(token) == 1 and token.isalpha()


def expand_initials(tokens: list[str]) -> list[str]:
    """Return tokens with initials stripped of trailing period."""
    return [t.rstrip(".") for t in tokens]


def sort_tokens(tokens: list[str]) -> list[str]:
    """Alphabetically sort tokens — used for person names only."""
    return sorted(tokens)


def remove_extra_spaces(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()
