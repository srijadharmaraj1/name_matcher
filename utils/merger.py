"""
merger.py — Merge match results back to full dataframes.
Produces three datasets:
  1. Matched — df1 rows + df2 matched rows + score/category/reason (vertical, one row per pair)
  2. Only in DF1 — df1 rows with no match in df2
  3. Only in DF2 — df2 rows never matched by any df1 row
"""

import pandas as pd


def build_output_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_results: dict,
    assembled_col: str = "_assembled_name",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three output dataframes from match results.

    Args:
        df1: full original dataframe 1
        df2: full original dataframe 2
        match_results: {name_a: [result_dict, ...], ...}
        assembled_col: name of the assembled name column (to drop from output)

    Returns:
        (matched_df, only_df1_df, only_df2_df)
    """
    # Prefix columns to avoid collision
    df1_clean = df1.drop(columns=[assembled_col], errors="ignore").copy()
    df2_clean = df2.drop(columns=[assembled_col], errors="ignore").copy()

    df1_clean.columns = [f"DF1_{c}" for c in df1_clean.columns]
    df2_clean.columns = [f"DF2_{c}" for c in df2_clean.columns]

    # Build name → rows maps
    df1_name_map = _build_name_map(df1, assembled_col)
    df2_name_map = _build_name_map(df2, assembled_col)

    matched_rows = []
    matched_df1_names = set()
    matched_df2_names = set()

    match_number_tracker = {}

    for name_a, matches in match_results.items():
        if not matches:
            continue

        matched_df1_names.add(name_a)
        df1_row_indices = df1_name_map.get(name_a, [])

        for match in matches:
            name_b = match["name_b"]
            matched_df2_names.add(name_b)
            df2_row_indices = df2_name_map.get(name_b, [])

            match_number_tracker[name_a] = match_number_tracker.get(name_a, 0) + 1
            match_num = match_number_tracker[name_a]

            score = round(match["score"], 2)
            category = match["category"]
            reason = "; ".join(match["reasons"])
            llm_used = match.get("llm_used", False)
            llm_confidence = match.get("llm_confidence", "")

            # Cross-product: each df1 row × each df2 matched row
            for i1 in df1_row_indices:
                for i2 in df2_row_indices:
                    row = {}
                    row.update(df1_clean.loc[i1].to_dict())
                    row.update(df2_clean.loc[i2].to_dict())
                    row["Match_Number"] = match_num
                    row["Score"] = score
                    row["Category"] = category
                    row["Reason"] = reason
                    row["LLM_Used"] = llm_used
                    row["LLM_Confidence"] = llm_confidence if llm_used else ""
                    matched_rows.append(row)

    # Only in DF1 — names that got no match
    only_df1_rows = []
    for name_a, matches in match_results.items():
        if not matches:
            df1_row_indices = df1_name_map.get(name_a, [])
            for i1 in df1_row_indices:
                only_df1_rows.append(df1_clean.loc[i1].to_dict())

    # Only in DF2 — names in df2 never matched
    all_df2_names = set(df2_name_map.keys())
    unmatched_df2_names = all_df2_names - matched_df2_names
    only_df2_rows = []
    for name_b in unmatched_df2_names:
        df2_row_indices = df2_name_map.get(name_b, [])
        for i2 in df2_row_indices:
            only_df2_rows.append(df2_clean.loc[i2].to_dict())

    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
    only_df1_df = pd.DataFrame(only_df1_rows) if only_df1_rows else pd.DataFrame()
    only_df2_df = pd.DataFrame(only_df2_rows) if only_df2_rows else pd.DataFrame()

    return matched_df, only_df1_df, only_df2_df


def _build_name_map(df: pd.DataFrame, assembled_col: str) -> dict:
    """Map assembled name → list of row indices."""
    mapping = {}
    for idx, row in df.iterrows():
        name = row.get(assembled_col, "")
        if name and str(name).strip():
            mapping.setdefault(name, []).append(idx)
    return mapping
