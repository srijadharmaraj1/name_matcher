"""
deduper.py — Extract unique assembled names from a dataframe.
Matching runs on unique names only for performance.
Results are merged back to full data after matching.
"""

import pandas as pd


def extract_unique_names(df: pd.DataFrame, assembled_col: str = "_assembled_name") -> list[str]:
    """
    Extract sorted list of unique non-empty assembled names.
    """
    names = df[assembled_col].dropna().unique().tolist()
    return [n for n in names if n and str(n).strip()]


def get_name_to_rows_map(df: pd.DataFrame, assembled_col: str = "_assembled_name") -> dict:
    """
    Map each unique name to its row indices in the dataframe.
    Used to merge results back after matching.

    Returns: {name: [idx1, idx2, ...], ...}
    """
    mapping = {}
    for idx, row in df.iterrows():
        name = row.get(assembled_col, "")
        if name and str(name).strip():
            mapping.setdefault(name, []).append(idx)
    return mapping
