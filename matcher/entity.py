"""
entity.py — Entity-specific matching logic.
Handles stop word stripping, core token extraction, and core-only matching.
Key rule: stop words stripped for matching only, never for display.
"""

from pathlib import Path
import yaml
from matcher.normalizer import normalize, tokenize


_CONFIG_DIR = Path(__file__).parent.parent / "config"
_stopwords_cache = None
_particles_cache = None


def _load_stopwords() -> set:
    global _stopwords_cache
    if _stopwords_cache is None:
        path = _CONFIG_DIR / "stopwords.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        _stopwords_cache = set(w.lower() for w in data.get("entity", []))
    return _stopwords_cache


def _load_particles() -> set:
    global _particles_cache
    if _particles_cache is None:
        path = _CONFIG_DIR / "particles.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        particles = data.get("particles", [])
        titles = data.get("person_titles", [])
        suffixes = data.get("person_suffixes", [])
        _particles_cache = set(w.lower() for w in particles + titles + suffixes)
    return _particles_cache


def extract_core_tokens(name: str) -> list[str]:
    """
    Strip entity stop words from a normalized name.
    Returns core tokens only (the meaningful part of the entity name).
    """
    stopwords = _load_stopwords()
    tokens = tokenize(normalize(name))
    core = [t for t in tokens if t not in stopwords]
    return core if core else tokens  # fallback to all tokens if all stripped


def strip_person_noise(name: str) -> list[str]:
    """
    Strip titles, suffixes, and particles from person name tokens.
    Returns cleaned tokens.
    """
    noise = _load_particles()
    tokens = tokenize(normalize(name))
    cleaned = [t for t in tokens if t not in noise]
    return cleaned if cleaned else tokens


def core_token_exact_match(core_a: list[str], core_b: list[str]) -> float:
    """
    Exact match on sorted core tokens.
    Returns 1.0 if identical, 0.0 otherwise.
    """
    if not core_a or not core_b:
        return 0.0
    return 1.0 if sorted(core_a) == sorted(core_b) else 0.0


def core_token_overlap(core_a: list[str], core_b: list[str]) -> float:
    """
    Token overlap ratio between two core token lists.
    Handles partial entity name matches.
    Returns 0.0–1.0.
    """
    if not core_a or not core_b:
        return 0.0
    set_a = set(core_a)
    set_b = set(core_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def is_stopword_only_match(name_a: str, name_b: str) -> bool:
    """
    Returns True if two names share ONLY stop words and no core tokens.
    e.g. 'ABC Ltd' vs 'XYZ Ltd' → cores are different → False (not stopword-only)
    e.g. 'Ltd Co' vs 'Inc Ltd' → cores empty → True (stopword-only, no real match)
    """
    core_a = extract_core_tokens(name_a)
    core_b = extract_core_tokens(name_b)
    all_a = set(tokenize(normalize(name_a)))
    all_b = set(tokenize(normalize(name_b)))
    stopwords = _load_stopwords()

    shared = all_a & all_b
    shared_core = set(core_a) & set(core_b)

    # All shared tokens are stop words → stopword-only match
    return len(shared) > 0 and all(t in stopwords for t in shared) and len(shared_core) == 0
