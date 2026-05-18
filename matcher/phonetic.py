"""
phonetic.py — Multi-algorithm phonetic matching.
Algorithms: Soundex, Double Metaphone (via jellyfish), Caverphone.
Voting: 2 out of 3 must agree for a phonetic match.
Also includes Jaro-Winkler for prefix-sensitive short name comparison.
"""

import re


# ─── SOUNDEX ─────────────────────────────────────────────────────────────────

def soundex(name: str) -> str:
    """Standard Soundex algorithm."""
    if not name:
        return "0000"

    name = name.upper()
    name = re.sub(r"[^A-Z]", "", name)
    if not name:
        return "0000"

    codes = {
        "BFPV": "1", "CGJKQSXYZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6"
    }

    def get_code(char):
        for key, val in codes.items():
            if char in key:
                return val
        return "0"

    first = name[0]
    encoded = first
    prev_code = get_code(first)

    for char in name[1:]:
        code = get_code(char)
        if code != "0" and code != prev_code:
            encoded += code
        prev_code = code

    encoded = encoded[:4].ljust(4, "0")
    return encoded


# ─── DOUBLE METAPHONE ────────────────────────────────────────────────────────

def double_metaphone(name: str) -> tuple[str, str]:
    """
    Double Metaphone — returns (primary, secondary) codes.
    Uses jellyfish if available, falls back to simple implementation.
    """
    try:
        import jellyfish
        primary = jellyfish.metaphone(name)
        return (primary, primary)
    except ImportError:
        pass

    # Simple Metaphone fallback
    if not name:
        return ("", "")

    name = name.upper()
    name = re.sub(r"[^A-Z]", "", name)
    if not name:
        return ("", "")

    replacements = [
        ("PH", "F"), ("CK", "K"), ("QU", "K"), ("GN", "N"),
        ("KN", "N"), ("WR", "R"), ("SCH", "SK"), ("TH", "0"),
        ("TCH", "X"), ("SH", "X"),
    ]

    result = name
    for old, new in replacements:
        result = result.replace(old, new)

    # Remove duplicates
    deduped = result[0] if result else ""
    for i in range(1, len(result)):
        if result[i] != result[i - 1]:
            deduped += result[i]

    # Remove vowels except first
    if deduped:
        cleaned = deduped[0] + re.sub(r"[AEIOU]", "", deduped[1:])
    else:
        cleaned = ""

    return (cleaned, cleaned)


# ─── CAVERPHONE ──────────────────────────────────────────────────────────────

def caverphone(name: str) -> str:
    """
    Caverphone 2.0 — good for English names, stricter than Metaphone.
    Uses jellyfish if available, falls back to simplified version.
    """
    try:
        import jellyfish
        return jellyfish.caverphone(name)
    except (ImportError, AttributeError):
        pass

    # Simplified Caverphone fallback
    if not name:
        return ""

    w = name.lower()
    w = re.sub(r"[^a-z]", "", w)

    replacements = [
        ("cough", "cof"), ("rough", "rof"), ("tough", "tof"),
        ("enough", "enof"), ("trough", "trof"),
        ("ph", "f"), ("qu", "k"), ("ck", "k"),
        ("ae", "a"), ("gh", "k"),
        ("mb", "m"), ("cq", "k"),
    ]

    for old, new in replacements:
        w = w.replace(old, new)

    # Remove vowels after first char
    if w:
        w = w[0] + re.sub(r"[aeiou]", "3", w[1:])

    w = re.sub(r"3+", "3", w)
    w = re.sub(r"3", "", w)

    return w.upper()[:10].ljust(10, "1")


# ─── JARO-WINKLER ────────────────────────────────────────────────────────────

def jaro_winkler(s1: str, s2: str) -> float:
    """
    Jaro-Winkler similarity — prefix-sensitive, good for short names and initials.
    Returns 0.0–1.0.
    """
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(s1, s2)
    except ImportError:
        pass

    # Pure Python Jaro-Winkler
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    match_dist = max(0, match_dist)

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


# ─── PHONETIC MATCH ──────────────────────────────────────────────────────────

def phonetic_match(token_a: str, token_b: str, vote_threshold: int = 2) -> tuple[bool, list[str]]:
    """
    Run all 3 phonetic algorithms on a token pair.
    Returns (match: bool, algorithms_that_matched: list).
    vote_threshold: how many must agree (default 2 of 3).
    """
    matched = []

    # Soundex
    if soundex(token_a) == soundex(token_b):
        matched.append("soundex")

    # Double Metaphone
    meta_a = double_metaphone(token_a)
    meta_b = double_metaphone(token_b)
    if meta_a[0] and meta_b[0] and (
        meta_a[0] == meta_b[0] or meta_a[1] == meta_b[0] or
        meta_a[0] == meta_b[1] or meta_a[1] == meta_b[1]
    ):
        matched.append("metaphone")

    # Caverphone
    if caverphone(token_a) == caverphone(token_b):
        matched.append("caverphone")

    return (len(matched) >= vote_threshold, matched)


def phonetic_score_tokens(tokens_a: list[str], tokens_b: list[str], vote_threshold: int = 2) -> float:
    """
    Compute phonetic match score between two token lists.
    Matches each token in A against best phonetic match in B.
    Returns 0.0–1.0.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    matched_count = 0
    for ta in tokens_a:
        if len(ta) <= 1:
            continue  # skip initials for phonetic
        for tb in tokens_b:
            if len(tb) <= 1:
                continue
            match, _ = phonetic_match(ta, tb, vote_threshold)
            if match:
                matched_count += 1
                break

    total = max(len([t for t in tokens_a if len(t) > 1]), 1)
    return matched_count / total
