"""
scorer.py — Weighted multi-signal score combiner.
Combines: exact token, fuzzy, phonetic, structural, jaro-winkler.
Different weights for person vs entity.
Returns final score (0–100), category, and list of reason strings.
"""

from matcher.normalizer import tokenize, normalize
from matcher.fuzzy import fuzzy_score
from matcher.phonetic import phonetic_score_tokens, jaro_winkler
from matcher.structural import structural_score
from matcher.entity import extract_core_tokens, core_token_exact_match, core_token_overlap, is_stopword_only_match


def _exact_token_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Sorted token set exact match."""
    if not tokens_a or not tokens_b:
        return 0.0
    return 1.0 if sorted(tokens_a) == sorted(tokens_b) else 0.0


def score_pair(
    name_a: str,
    name_b: str,
    name_type: str,
    weights: dict,
    config: dict,
) -> dict:
    """
    Score a name pair and return full result dict.

    Returns:
        {
            name_a, name_b, name_type,
            score (0–100),
            category (Exact/High/Medium/Low/No Match),
            reasons (list of strings),
            signals (dict of raw signal scores),
            needs_llm (bool)
        }
    """
    reasons = []
    signals = {}

    norm_a = normalize(name_a)
    norm_b = normalize(name_b)

    # ── ENTITY PATH ──────────────────────────────────────────────────────────
    if name_type == "entity":
        # Guard: stopword-only match → no match
        if is_stopword_only_match(name_a, name_b):
            return _build_result(
                name_a, name_b, name_type, 0.0,
                ["Only stop words shared — no core token match"],
                {}, config
            )

        core_a = extract_core_tokens(name_a)
        core_b = extract_core_tokens(name_b)

        # Exact core match
        exact = core_token_exact_match(core_a, core_b)
        signals["exact_token"] = exact
        if exact == 1.0:
            reasons.append("Core entity tokens match exactly")

        # Fuzzy on core
        core_str_a = " ".join(sorted(core_a))
        core_str_b = " ".join(sorted(core_b))
        fuzz, fuzz_method = fuzzy_score(core_str_a, core_str_b)
        signals["fuzzy"] = fuzz
        if fuzz > 0.7:
            reasons.append(f"Fuzzy match on core tokens ({fuzz_method}, {fuzz*100:.0f}%)")

        # Phonetic on core
        phon = phonetic_score_tokens(core_a, core_b, config.get("phonetic_vote_threshold", 2))
        signals["phonetic"] = phon
        if phon > 0.5:
            reasons.append(f"Phonetic match on core tokens ({phon*100:.0f}%)")

        # Structural on core
        struct, struct_signals = structural_score(
            core_a, core_b,
            initial_max_score=config.get("initial_match_max_score", 70) / 100
        )
        signals["structural"] = struct
        if struct_signals:
            reasons.append(f"Structural: {', '.join(struct_signals)}")

        # Jaro-Winkler on core string
        jw = jaro_winkler(core_str_a, core_str_b)
        signals["jaro_winkler"] = jw
        if jw > 0.85:
            reasons.append(f"Jaro-Winkler similarity ({jw*100:.0f}%)")

    # ── PERSON PATH ──────────────────────────────────────────────────────────
    else:
        tokens_a = tokenize(norm_a)
        tokens_b = tokenize(norm_b)

        # Exact token set
        exact = _exact_token_score(tokens_a, tokens_b)
        signals["exact_token"] = exact
        if exact == 1.0:
            reasons.append("Exact token match (order-normalized)")

        # Fuzzy
        fuzz, fuzz_method = fuzzy_score(norm_a, norm_b)
        signals["fuzzy"] = fuzz
        if fuzz > 0.7:
            reasons.append(f"Fuzzy match ({fuzz_method}, {fuzz*100:.0f}%)")

        # Phonetic
        phon = phonetic_score_tokens(tokens_a, tokens_b, config.get("matching", {}).get("phonetic_vote_threshold", 1))
        signals["phonetic"] = phon
        if phon > 0.5:
            reasons.append(f"Phonetic match ({phon*100:.0f}%)")

        # Structural (initials, missing middle, subset)
        initial_cap = config.get("initial_match_max_score", 70) / 100
        struct, struct_signals = structural_score(tokens_a, tokens_b, initial_max_score=initial_cap)
        signals["structural"] = struct
        if struct_signals:
            reasons.append(f"Structural: {', '.join(struct_signals)}")

        # Jaro-Winkler
        jw = jaro_winkler(norm_a, norm_b)
        signals["jaro_winkler"] = jw
        if jw > 0.85:
            reasons.append(f"Jaro-Winkler similarity ({jw*100:.0f}%)")

    # ── WEIGHTED SCORE ───────────────────────────────────────────────────────
    w = weights
    raw_score = (
        signals.get("exact_token", 0.0) * w.get("exact_token", 0.30) +
        signals.get("fuzzy", 0.0)       * w.get("fuzzy", 0.25) +
        signals.get("phonetic", 0.0)    * w.get("phonetic", 0.25) +
        signals.get("structural", 0.0)  * w.get("structural", 0.10) +
        signals.get("jaro_winkler", 0.0)* w.get("jaro_winkler", 0.10)
    )

    final_score = round(raw_score * 100, 2)

    if not reasons:
        reasons.append("No strong signals found")

    return _build_result(name_a, name_b, name_type, final_score, reasons, signals, config)


def _build_result(
    name_a: str,
    name_b: str,
    name_type: str,
    score: float,
    reasons: list[str],
    signals: dict,
    config: dict,
) -> dict:
    thresholds = config.get("thresholds", {})
    llm_lower = thresholds.get("llm_band_lower", 35)
    llm_upper = thresholds.get("llm_band_upper", 90)

    category = _categorize(score, thresholds)
    needs_llm = llm_lower <= score <= llm_upper

    return {
        "name_a": name_a,
        "name_b": name_b,
        "name_type": name_type,
        "score": score,
        "category": category,
        "reasons": reasons,
        "signals": signals,
        "needs_llm": needs_llm,
    }


def _categorize(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("exact", 95):
        return "Exact"
    elif score >= thresholds.get("high", 80):
        return "High"
    elif score >= thresholds.get("medium", 60):
        return "Medium"
    elif score >= thresholds.get("low", 40):
        return "Low"
    else:
        return "No Match"
