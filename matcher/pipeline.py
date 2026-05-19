"""
pipeline.py — Orchestrates the full name matching pipeline.
Flow: blocking → rule-based scoring → LLM escalation → results.
"""

from matcher.normalizer import tokenize, normalize
from matcher.scorer import score_pair, _categorize
from matcher.llm import call_llm
from matcher.phonetic import soundex


# ─── BLOCKING ────────────────────────────────────────────────────────────────

def _blocking_key(name: str, prefix_len: int = 3) -> set:
    """
    Generate blocking keys for a name to reduce candidate pairs.
    Returns set of keys — pairs sharing any key are compared.
    Keys: first N chars of each token, soundex of each token.
    """
    keys = set()
    tokens = tokenize(normalize(name))
    for token in tokens:
        if len(token) >= prefix_len:
            keys.add(token[:prefix_len])
        if len(token) > 1:
            keys.add(f"sdx_{soundex(token)}")
    return keys


def _should_compare(name_a: str, name_b: str, prefix_len: int = 3) -> bool:
    """Return True if two names share at least one blocking key."""
    keys_a = _blocking_key(name_a, prefix_len)
    keys_b = _blocking_key(name_b, prefix_len)
    return bool(keys_a & keys_b)


# ─── SINGLE PAIR ─────────────────────────────────────────────────────────────

def match_pair(
    name_a: str,
    name_b: str,
    name_type: str,
    weights: dict,
    config: dict,
    llm_config: dict = None,
) -> dict:
    """
    Match a single name pair through the full pipeline.
    Returns complete result dict.
    """
    thresholds = config.get("thresholds", {})
    llm_lower = thresholds.get("llm_band_lower", 35)
    llm_upper = thresholds.get("llm_band_upper", 90)

    # Rule-based score
    result = score_pair(name_a, name_b, name_type, weights, config)

    # LLM escalation for ambiguous band
    if result["needs_llm"] and llm_config and llm_config.get("enabled", False):
        llm_result = call_llm(
            name_a=name_a,
            name_b=name_b,
            name_type=name_type,
            signals=result["signals"],
            rule_score=result["score"],
            deployment=llm_config.get("deployment"),
            api_version=llm_config.get("api_version"),
            max_tokens=config.get("llm", {}).get("max_tokens", 500),
            temperature=config.get("llm", {}).get("temperature", 0),
        )

        # Update result with LLM output
        result["score"] = float(llm_result["adjusted_score"])
        result["category"] = _categorize(result["score"], thresholds)
        result["reasons"].append(f"LLM: {llm_result['reason']} (confidence: {llm_result['confidence']}%)")
        result["llm_used"] = llm_result["llm_used"]
        result["llm_confidence"] = llm_result["confidence"]
    else:
        result["llm_used"] = False
        result["llm_confidence"] = None

    return result


# ─── BATCH MATCHING ──────────────────────────────────────────────────────────

def match_unique_names(
    unique_a: list[str],
    unique_b: list[str],
    name_type: str,
    weights: dict,
    config: dict,
    llm_config: dict = None,
    progress_callback=None,
) -> dict:
    """
    Match two lists of unique names.
    Uses blocking to reduce candidate pairs.
    Returns dict: {name_a: [result, result, ...], ...}

    progress_callback: optional callable(current, total) for progress updates.
    """
    thresholds = config.get("thresholds", {})
    no_match_threshold = thresholds.get("llm_band_lower", 35)
    max_matches = config.get("matching", {}).get("max_matches_per_name", 5)
    prefix_len = config.get("matching", {}).get("blocking_prefix_length", 3)

    results = {name: [] for name in unique_a}
    total = len(unique_a)

    for i, name_a in enumerate(unique_a):
        if progress_callback:
            progress_callback(i + 1, total)

        candidates = []
        for name_b in unique_b:
            # Blocking check — skip if no shared keys
            if not _should_compare(name_a, name_b, prefix_len):
                continue

            result = match_pair(name_a, name_b, name_type, weights, config, llm_config)

            if result["score"] >= no_match_threshold:
                candidates.append(result)

        # Sort by score descending, take top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results[name_a] = candidates[:max_matches]

    return results


# ─── STATS ───────────────────────────────────────────────────────────────────

def compute_summary(
    match_results: dict,
    df1_total: int,
    df2_total: int,
    unique_b: list[str],
) -> dict:
    """
    Compute summary statistics from match results.
    Returns dict with counts for dashboard display.
    """
    matched_names_a = set()
    matched_names_b = set()
    category_counts = {"Exact": 0, "High": 0, "Medium": 0, "Low": 0}
    all_scores = []
    llm_escalations = 0
    total_pairs = 0

    for name_a, matches in match_results.items():
        if matches:
            matched_names_a.add(name_a)
            for m in matches:
                matched_names_b.add(m["name_b"])
                cat = m.get("category", "No Match")
                if cat in category_counts:
                    category_counts[cat] += 1
                all_scores.append(m["score"])
                if m.get("llm_used"):
                    llm_escalations += 1
                total_pairs += 1

    only_in_a = set(match_results.keys()) - matched_names_a
    only_in_b = set(unique_b) - matched_names_b

    return {
        "df1_total": df1_total,
        "df2_total": df2_total,
        "unique_a": len(match_results),
        "unique_b": len(unique_b),
        "matched_pairs": total_pairs,
        "matched_names_a": len(matched_names_a),
        "matched_names_b": len(matched_names_b),
        "only_in_a": len(only_in_a),
        "only_in_b": len(only_in_b),
        "category_counts": category_counts,
        "avg_score": round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
        "llm_escalations": llm_escalations,
        "total_comparisons": total_pairs,
    }