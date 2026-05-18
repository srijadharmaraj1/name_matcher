"""
fuzzy.py — Fuzzy string matching using edit distance and token-based methods.
Uses rapidfuzz if available, falls back to pure Python difflib.
Methods: Edit Distance ratio, Token Sort Ratio, Token Set Ratio.
Final score = max of all three methods.
"""

import difflib


def _edit_ratio(s1: str, s2: str) -> float:
    """Basic edit distance ratio via difflib SequenceMatcher."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def _token_sort_ratio(s1: str, s2: str) -> float:
    """Sort tokens before comparing — handles word order differences."""
    s1_sorted = " ".join(sorted(s1.split()))
    s2_sorted = " ".join(sorted(s2.split()))
    return _edit_ratio(s1_sorted, s2_sorted)


def _token_set_ratio(s1: str, s2: str) -> float:
    """
    Compare intersection and remainders of token sets.
    Handles subset names: 'John Smith' vs 'John Andrew Smith'.
    """
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())

    intersection = tokens1 & tokens2
    remainder1 = tokens1 - tokens2
    remainder2 = tokens2 - tokens1

    t0 = " ".join(sorted(intersection))
    t1 = " ".join(sorted(intersection | remainder1))
    t2 = " ".join(sorted(intersection | remainder2))

    scores = [
        _edit_ratio(t0, t1),
        _edit_ratio(t0, t2),
        _edit_ratio(t1, t2),
    ]
    return max(scores)


def fuzzy_score(s1: str, s2: str) -> tuple[float, str]:
    """
    Compute fuzzy score between two strings.
    Returns (score 0.0–1.0, method_used).
    Uses rapidfuzz if available for speed, falls back to difflib.
    Final = max of edit ratio, token sort ratio, token set ratio.
    """
    if not s1 and not s2:
        return (1.0, "exact_empty")
    if not s1 or not s2:
        return (0.0, "empty")

    # Try rapidfuzz first
    try:
        from rapidfuzz import fuzz
        edit = fuzz.ratio(s1, s2) / 100.0
        sort = fuzz.token_sort_ratio(s1, s2) / 100.0
        tset = fuzz.token_set_ratio(s1, s2) / 100.0
    except ImportError:
        edit = _edit_ratio(s1, s2)
        sort = _token_sort_ratio(s1, s2)
        tset = _token_set_ratio(s1, s2)

    scores = {"edit_ratio": edit, "token_sort": sort, "token_set": tset}
    best_method = max(scores, key=scores.get)
    return (scores[best_method], best_method)


def fuzzy_score_tokens(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Token-level fuzzy matching.
    For each token in A, find best fuzzy match in B.
    Returns average of best matches.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    total = 0.0
    for ta in tokens_a:
        best = max(fuzzy_score(ta, tb)[0] for tb in tokens_b)
        total += best

    return total / len(tokens_a)
