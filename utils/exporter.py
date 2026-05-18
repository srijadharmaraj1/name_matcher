"""
exporter.py — Export results to Excel (3 sheets) and optional CSV files.
Sheet 1: Matched
Sheet 2: Only in DF1
Sheet 3: Only in DF2
"""

import pandas as pd
from pathlib import Path
import io


def export_to_excel(
    matched_df: pd.DataFrame,
    only_df1_df: pd.DataFrame,
    only_df2_df: pd.DataFrame,
    output_path: str = None,
) -> bytes:
    """
    Write three dataframes to a single Excel file with 3 sheets.
    If output_path is given, writes to disk.
    Always returns bytes (for Streamlit download).
    """
    buffer = io.BytesIO()
    target = output_path if output_path else buffer

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        _write_sheet(writer, matched_df, "Matched")
        _write_sheet(writer, only_df1_df, "Only in DF1")
        _write_sheet(writer, only_df2_df, "Only in DF2")

    if output_path:
        # Also fill buffer for return
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            _write_sheet(writer, matched_df, "Matched")
            _write_sheet(writer, only_df1_df, "Only in DF1")
            _write_sheet(writer, only_df2_df, "Only in DF2")

    buffer.seek(0)
    return buffer.getvalue()


def _write_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str):
    """Write dataframe to Excel sheet with auto column widths."""

    if df.empty:
        pd.DataFrame(
            {"Info": [f"No records in '{sheet_name}'"]}
        ).to_excel(writer, sheet_name=sheet_name, index=False)
        return

    # Convert problematic Arrow/string dtypes to plain Python strings
    safe_df = df.copy()

    for col in safe_df.columns:
        safe_df[col] = safe_df[col].astype(str)

    safe_df.to_excel(writer, sheet_name=sheet_name, index=False)

    worksheet = writer.sheets[sheet_name]

    # Auto-fit column widths safely
    for col_idx, col in enumerate(safe_df.columns, 1):

        values = safe_df[col].fillna("").astype(str)

        max_len = max(
            len(str(col)),
            values.str.len().max() if not values.empty else 0,
        )

        col_letter = worksheet.cell(
            row=1,
            column=col_idx
        ).column_letter

        worksheet.column_dimensions[col_letter].width = min(
            max_len + 4,
            50,
        )


def export_csv(df: pd.DataFrame) -> bytes:
    """Export a single dataframe to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def ensure_output_dir(path: str):
    """Create output directory if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
