"""
structural.py — Structural name matching signals.
Handles: initial expansion, missing middle name, token subset, order variants.
Key rule: initial-only matches are capped at max_initial_score (never Exact/High alone).
"""

from matcher.normalizer import tokenize, is_initial


def initial_match_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Check if initials in one name match full tokens in the other.
    e.g. ['j', 'smith'] vs ['john', 'smith'] → matches on 'j' → 'john'

    Returns 0.0–1.0 raw score (caller applies cap).
    Higher score if more tokens match, lower if only initials match.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Determine which list has initials
    initials_a = [t for t in tokens_a if is_initial(t)]
    initials_b = [t for t in tokens_b if is_initial(t)]

    has_initials = len(initials_a) > 0 or len(initials_b) > 0
    if not has_initials:
        return 0.0

    # Count how many initials match a full token in the other list
    matched = 0
    total_initials = 0

    for ta in tokens_a:
        if is_initial(ta):
            total_initials += 1
            for tb in tokens_b:
                if not is_initial(tb) and tb.startswith(ta):
                    matched += 1
                    break

    for tb in tokens_b:
        if is_initial(tb):
            total_initials += 1
            for ta in tokens_a:
                if not is_initial(ta) and ta.startswith(tb):
                    matched += 1
                    break

    if total_initials == 0:
        return 0.0

    return matched / total_initials


def missing_middle_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Score names where one has a middle name/token the other lacks.
    e.g. ['john', 'andrew', 'smith'] vs ['john', 'smith'] → high score

    Returns 0.0–1.0.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Find common tokens
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    shared = set_a & set_b

    if not shared:
        return 0.0

    # Score based on what fraction of the shorter name is covered
    shorter = min(len(tokens_a), len(tokens_b))
    return len(shared) / shorter


def token_subset_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Check if one token list is a subset of the other.
    Returns 1.0 for full subset, partial score for overlap.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if set_a <= set_b or set_b <= set_a:
        return 1.0

    # Partial overlap
    intersection = set_a & set_b
    shorter = min(len(set_a), len(set_b))
    return len(intersection) / shorter if shorter > 0 else 0.0


def structural_score(
    tokens_a: list[str],
    tokens_b: list[str],
    initial_max_score: float = 0.70
) -> tuple[float, list[str]]:
    """
    Combined structural score.
    Returns (score 0.0–1.0, list of signals that fired).

    Signals checked:
    - initial_match: one name has initials matching other's full tokens
    - missing_middle: one name has extra middle token
    - token_subset: one name's tokens are subset of other
    """
    signals = []
    scores = []

    # Initial match
    init_score = initial_match_score(tokens_a, tokens_b)
    if init_score > 0:
        capped = min(init_score, initial_max_score)
        signals.append(f"initial_match({init_score:.2f}→capped@{initial_max_score})")
        scores.append(capped)

    # Missing middle
    mid_score = missing_middle_score(tokens_a, tokens_b)
    if mid_score > 0.5:
        signals.append(f"missing_middle({mid_score:.2f})")
        scores.append(mid_score)

    # Token subset
    sub_score = token_subset_score(tokens_a, tokens_b)
    if sub_score > 0.5 and "missing_middle" not in str(signals):
        signals.append(f"token_subset({sub_score:.2f})")
        scores.append(sub_score)

    if not scores:
        return (0.0, [])

    return (max(scores), signals)
