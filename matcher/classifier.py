"""
classifier.py — Auto-detect whether a name is a Person or Entity.
Used when name_type is not explicitly set (future mixed mode).
For now also used to validate/confirm user-provided type.
"""

import re
from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_entity_stopwords() -> set:
    data = _load_yaml("stopwords.yaml")
    return set(w.lower() for w in data.get("entity", []))


def _get_person_titles() -> set:
    data = _load_yaml("particles.yaml")
    return set(w.lower() for w in data.get("person_titles", []))


def classify(name: str) -> tuple[str, float]:
    """
    Classify a name as 'person' or 'entity'.
    Returns (type, confidence) where confidence is 0.0–1.0.

    Rules (in priority order):
    1. Contains entity stop word → entity (high confidence)
    2. Contains person title → person (high confidence)
    3. Contains digits or legal patterns → entity
    4. Token count + structure heuristics → person or entity (medium confidence)
    5. Default → entity (low confidence)
    """
    if not name:
        return ("entity", 0.5)

    entity_words = _get_entity_stopwords()
    person_titles = _get_person_titles()

    normalized = name.lower().strip()
    tokens = [t.strip(".,") for t in normalized.split()]

    # Rule 1 — entity stop word found
    for token in tokens:
        if token in entity_words:
            return ("entity", 0.92)

    # Rule 2 — person title found
    if tokens and tokens[0] in person_titles:
        return ("person", 0.92)

    # Rule 3 — contains digits or & symbol → entity
    if re.search(r"\d", name) or "&" in name:
        return ("entity", 0.85)

    # Rule 4 — heuristics on token count and structure
    token_count = len(tokens)

    if token_count == 1:
        # Single token — could be brand or last name only → entity more likely
        return ("entity", 0.60)

    if token_count == 2:
        # Two tokens — likely person (First Last)
        all_alpha = all(t.isalpha() or (len(t) == 1) for t in tokens)
        if all_alpha:
            return ("person", 0.75)
        return ("entity", 0.60)

    if token_count == 3:
        # Three tokens — could be either
        all_alpha = all(t.isalpha() or t.endswith(".") for t in tokens)
        if all_alpha:
            return ("person", 0.65)
        return ("entity", 0.60)

    if token_count >= 4:
        # Many tokens — more likely entity
        return ("entity", 0.65)

    # Default
    return ("entity", 0.50)


def is_low_confidence(confidence: float, threshold: float = 0.70) -> bool:
    """Flag names where classification confidence is below threshold."""
    return confidence < threshold
