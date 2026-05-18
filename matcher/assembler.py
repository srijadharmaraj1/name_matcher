"""
assembler.py — Combine selected columns into a single full name string.
Handles: title, first, middle, last, suffix, full name column.
Person: assembles and token-sorts.
Entity: uses single column as-is.
"""

import pandas as pd
from matcher.normalizer import normalize, tokenize, sort_tokens, expand_initials, is_initial


def assemble_person_name(row: pd.Series, column_map: dict) -> str:
    """
    Assemble a person name from mapped columns.
    column_map keys: title, first, middle, last, suffix, full
    If 'full' is mapped, it overrides all others.
    Token-sorts the result for order-agnostic matching.
    """
    # Full name column takes priority
    if column_map.get("full") and column_map["full"] in row.index:
        raw = str(row[column_map["full"]] or "").strip()
        return _process_person_name(raw)

    parts = []
    for field in ["title", "first", "middle", "last", "suffix"]:
        col = column_map.get(field)
        if col and col in row.index:
            val = str(row[col] or "").strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(val)

    raw = " ".join(parts)
    return _process_person_name(raw)


def _process_person_name(raw: str) -> str:
    """Normalize, tokenize, expand initials, sort tokens for person name."""
    if not raw:
        return ""
    normalized = normalize(raw)
    tokens = tokenize(normalized)
    tokens = expand_initials(tokens)
    tokens = sort_tokens(tokens)
    return " ".join(tokens)


def assemble_entity_name(row: pd.Series, column_map: dict) -> str:
    """
    Assemble entity name — single column only, no sorting.
    column_map must have 'full' key.
    """
    col = column_map.get("full")
    if not col or col not in row.index:
        return ""
    raw = str(row[col] or "").strip()
    if not raw or raw.lower() in ("nan", "none"):
        return ""
    return normalize(raw)


def assemble_dataframe(
    df: pd.DataFrame,
    column_map: dict,
    name_type: str,
    assembled_col: str = "_assembled_name"
) -> pd.DataFrame:
    """
    Add assembled name column to dataframe.
    Returns dataframe with new column appended.
    """
    df = df.copy()

    if name_type == "person":
        df[assembled_col] = df.apply(
            lambda row: assemble_person_name(row, column_map), axis=1
        )
    else:
        df[assembled_col] = df.apply(
            lambda row: assemble_entity_name(row, column_map), axis=1
        )

    return df


def validate_column_map(df: pd.DataFrame, column_map: dict, name_type: str) -> list[str]:
    """
    Validate that mapped columns exist in dataframe.
    Returns list of error messages (empty if valid).
    """
    errors = []
    available = set(df.columns.tolist())

    if name_type == "entity":
        col = column_map.get("full")
        if not col:
            errors.append("Entity type requires 'full' column to be mapped.")
        elif col not in available:
            errors.append(f"Column '{col}' not found in dataframe. Available: {list(available)}")
        return errors

    # Person — need at least full OR (first + last)
    full_col = column_map.get("full")
    first_col = column_map.get("first")
    last_col = column_map.get("last")

    if full_col:
        if full_col not in available:
            errors.append(f"Column '{full_col}' not found in dataframe.")
    elif not first_col or not last_col:
        errors.append("Person type requires either 'full' column or both 'first' and 'last' columns.")
    else:
        for field in ["first", "last", "middle", "title", "suffix"]:
            col = column_map.get(field)
            if col and col not in available:
                errors.append(f"Column '{col}' not found in dataframe. Available: {list(available)}")

    return errors
