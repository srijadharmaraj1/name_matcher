"""
main.py — CLI entry point for the Name Matching Tool.
Reads config from config/config_cli.yaml and config/config_app.yaml.
Runs the full matching pipeline and writes results to Excel.

Usage:
    python main.py
    python main.py --config config/config_cli.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "config" / ".env")

from utils.config_loader import load_app_config, load_cli_config, get_weights, validate_cli_config
from utils.deduper import extract_unique_names
from utils.merger import build_output_dataframes
from utils.exporter import export_to_excel, ensure_output_dir
from matcher.assembler import assemble_dataframe, validate_column_map
from matcher.pipeline import match_unique_names, compute_summary
from matcher.llm import is_llm_configured


def print_banner():
    print("=" * 60)
    print("  Name Matching Tool — CLI Mode")
    print("=" * 60)


def print_summary(summary: dict):
    print("\n── Match Summary ──────────────────────────────────")
    print(f"  DF1 Total rows      : {summary['df1_total']}")
    print(f"  DF2 Total rows      : {summary['df2_total']}")
    print(f"  Unique names in DF1 : {summary['unique_a']}")
    print(f"  Unique names in DF2 : {summary['unique_b']}")
    print(f"\n  ✅ Matched pairs    : {summary['matched_pairs']}")
    print(f"  🔵 Only in DF1      : {summary['only_in_a']}")
    print(f"  🟡 Only in DF2      : {summary['only_in_b']}")
    print(f"\n  Category breakdown:")
    for cat, count in summary["category_counts"].items():
        bar = "█" * (count // max(summary["matched_pairs"] // 20, 1))
        print(f"    {cat:<10}: {count:>5}  {bar}")
    print(f"\n  Avg match score     : {summary['avg_score']}")
    print(f"  LLM escalations     : {summary['llm_escalations']}")
    print("──────────────────────────────────────────────────\n")


def progress_callback(current: int, total: int):
    pct = (current / total) * 100
    bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"\r  [{bar}] {pct:.0f}%  ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Name Matching Tool — CLI")
    parser.add_argument("--config", default="config/config_cli.yaml", help="Path to CLI config file")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading configuration...")
    try:
        app_config = load_app_config()
        cli_config = load_cli_config()
    except FileNotFoundError as e:
        print(f"\n❌ Config error: {e}")
        sys.exit(1)

    name_type = cli_config.get("name_type", "person")
    data_config = cli_config.get("data", {})
    df1_path = data_config.get("df1_path", "")
    df2_path = data_config.get("df2_path", "")
    output_path = data_config.get("output_path", "output/results.xlsx")
    df1_col_map = cli_config.get("df1_columns", {})
    df2_col_map = cli_config.get("df2_columns", {})

    print(f"  Name type : {name_type}")
    print(f"  DF1 path  : {df1_path}")
    print(f"  DF2 path  : {df2_path}")
    print(f"  Output    : {output_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[2/6] Loading data files...")
    try:
        df1 = pd.read_csv(df1_path) if df1_path.endswith(".csv") else pd.read_excel(df1_path)
        df2 = pd.read_csv(df2_path) if df2_path.endswith(".csv") else pd.read_excel(df2_path)
        print(f"  DF1: {len(df1)} rows, {len(df1.columns)} columns")
        print(f"  DF2: {len(df2)} rows, {len(df2.columns)} columns")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)

    # ── Validate config ───────────────────────────────────────────────────────
    errors = validate_cli_config(cli_config, df1, df2)
    if errors:
        print("\n❌ Configuration errors:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)

    # ── Assemble names ────────────────────────────────────────────────────────
    print("\n[3/6] Assembling names...")
    df1 = assemble_dataframe(df1, df1_col_map, name_type)
    df2 = assemble_dataframe(df2, df2_col_map, name_type)

    unique_a = extract_unique_names(df1)
    unique_b = extract_unique_names(df2)
    print(f"  Unique names in DF1: {len(unique_a)}")
    print(f"  Unique names in DF2: {len(unique_b)}")

    # ── LLM config ────────────────────────────────────────────────────────────
    llm_config = None
    if is_llm_configured():
        import os
        llm_config = {
            "enabled": True,
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        }
        print("\n  ✅ Azure LLM configured — ambiguous pairs will be escalated")
    else:
        print("\n  ⚠️  Azure LLM not configured — rule-based only (set .env to enable)")

    # ── Run matching ──────────────────────────────────────────────────────────
    weights = get_weights(app_config, name_type)
    print(f"\n[4/6] Running matching ({len(unique_a)} × {len(unique_b)} names)...")
    start = time.time()

    match_results = match_unique_names(
        unique_a=unique_a,
        unique_b=unique_b,
        name_type=name_type,
        weights=weights,
        config=app_config,
        llm_config=llm_config,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = compute_summary(match_results, len(df1), len(df2), unique_b)
    print_summary(summary)

    # ── Merge & export ────────────────────────────────────────────────────────
    print("[5/6] Merging results with original data...")
    matched_df, only_df1_df, only_df2_df = build_output_dataframes(df1, df2, match_results)

    print("[6/6] Exporting to Excel...")
    ensure_output_dir(output_path)
    export_to_excel(matched_df, only_df1_df, only_df2_df, output_path=output_path)

    print(f"\n✅ Done! Results saved to: {output_path}")
    print(f"   Sheet 1 — Matched      : {len(matched_df)} rows")
    print(f"   Sheet 2 — Only in DF1  : {len(only_df1_df)} rows")
    print(f"   Sheet 3 — Only in DF2  : {len(only_df2_df)} rows\n")


if __name__ == "__main__":
    main()
