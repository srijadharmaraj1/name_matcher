"""
app.py — Streamlit UI for the Name Matching Tool.
Pages: Setup → Preview → Results
"""

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "config" / ".env")

from matcher.assembler import assemble_dataframe, validate_column_map
from matcher.llm import is_llm_configured, get_models_from_config
from matcher.pipeline import match_unique_names, compute_summary
from utils.config_loader import load_app_config, get_weights
from utils.deduper import extract_unique_names
from utils.exporter import export_to_excel, export_csv
from utils.merger import build_output_dataframes

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Name Matcher",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "page": "setup",
    "df1": None,
    "df2": None,
    "df1_assembled": None,
    "df2_assembled": None,
    "match_results": None,
    "summary": None,
    "matched_df": None,
    "only_df1_df": None,
    "only_df2_df": None,
    "name_type": "person",
    "df1_col_map": {},
    "df2_col_map": {},
    "llm_config": {},
    "app_config": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def load_config():
    if st.session_state["app_config"] is None:
        st.session_state["app_config"] = load_app_config()
    return st.session_state["app_config"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_file(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    try:
        if uploaded.name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def column_selector(df: pd.DataFrame, label: str, col_map_key: str, name_type: str):
    """Render column mapping UI for a dataframe."""
    cols = ["— none —"] + df.columns.tolist()

    st.markdown(f"**{label}**")

    if name_type == "entity":
        st.caption("Entity mode: select exactly one column containing the full entity name.")
        full = st.selectbox("Full name column", cols, key=f"{col_map_key}_full")
        return {"full": None if full == "— none —" else full}

    else:
        st.caption("Select columns for each name part. Set 'Full name' if name is in one column.")
        col1, col2 = st.columns(2)
        with col1:
            full = st.selectbox("Full name column (overrides all below)", cols, key=f"{col_map_key}_full")
            first = st.selectbox("First name", cols, key=f"{col_map_key}_first")
            middle = st.selectbox("Middle name", cols, key=f"{col_map_key}_middle")
        with col2:
            last = st.selectbox("Last name", cols, key=f"{col_map_key}_last")
            title = st.selectbox("Title (Mr/Dr etc.)", cols, key=f"{col_map_key}_title")
            suffix = st.selectbox("Suffix (Jr/Sr etc.)", cols, key=f"{col_map_key}_suffix")

        def val(v):
            return None if v == "— none —" else v

        return {
            "full": val(full),
            "first": val(first),
            "middle": val(middle),
            "last": val(last),
            "title": val(title),
            "suffix": val(suffix),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def page_setup():
    st.title("🔤 Name Matching Tool")
    st.caption("Match person or entity names across two datasets with fuzzy, phonetic, and AI-powered matching.")

    config = load_config()

    # ── Type selection ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("1. Select Name Type")
    name_type = st.radio(
        "What type of names are you matching?",
        options=["person", "entity"],
        format_func=lambda x: "👤 Person names" if x == "person" else "🏢 Entity / Company names",
        horizontal=True,
        index=0 if st.session_state["name_type"] == "person" else 1,
    )
    st.session_state["name_type"] = name_type

    # ── Upload & map columns ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Upload & Map Columns")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### DataFrame 1")
        f1 = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="upload_df1")
        if f1:
            df1 = load_file(f1)
            if df1 is not None:
                st.session_state["df1"] = df1
                st.success(f"✅ {len(df1)} rows, {len(df1.columns)} columns")

        if st.session_state["df1"] is not None:
            st.session_state["df1_col_map"] = column_selector(
                st.session_state["df1"], "Map name columns — DF1", "df1", name_type
            )

    with col_right:
        st.markdown("#### DataFrame 2")
        f2 = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="upload_df2")
        if f2:
            df2 = load_file(f2)
            if df2 is not None:
                st.session_state["df2"] = df2
                st.success(f"✅ {len(df2)} rows, {len(df2.columns)} columns")

        if st.session_state["df2"] is not None:
            st.session_state["df2_col_map"] = column_selector(
                st.session_state["df2"], "Map name columns — DF2", "df2", name_type
            )

    # ── Match settings ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Match Settings")

    thresholds = config.get("thresholds", {})
    matching = config.get("matching", {})

    s1, s2, s3 = st.columns(3)
    with s1:
        llm_lower = st.number_input(
            "LLM band — lower threshold",
            min_value=0, max_value=100,
            value=thresholds.get("llm_band_lower", 35),
            help="Pairs scoring above this go to AI for review"
        )
    with s2:
        llm_upper = st.number_input(
            "LLM band — upper threshold",
            min_value=0, max_value=100,
            value=thresholds.get("llm_band_upper", 90),
            help="Pairs scoring above this are Exact — no AI needed"
        )
    with s3:
        max_matches = st.number_input(
            "Max matches per name",
            min_value=1, max_value=20,
            value=matching.get("max_matches_per_name", 5),
        )

    # Update config in session
    config["thresholds"]["llm_band_lower"] = llm_lower
    config["thresholds"]["llm_band_upper"] = llm_upper
    config["matching"]["max_matches_per_name"] = max_matches
    st.session_state["app_config"] = config

    # ── Azure LLM settings ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Azure LLM")

    credentials_present = is_llm_configured()

    use_azure = st.toggle(
        "Use Azure LLM for ambiguous matches",
        value=False,
        help="When enabled, name pairs scoring between the two thresholds are escalated to Azure OpenAI. Credentials are read from config/.env",
    )

    if use_azure:
        if not credentials_present:
            st.error(
                "Azure credentials not found in config/.env — "
                "set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_OPENAI_ENDPOINT."
            )
            use_azure = False
        else:
            models, default_model = get_models_from_config(config)
            model_options = [m["name"] for m in models]
            default_index = next(
                (i for i, m in enumerate(models) if m.get("default")), 0
            )
            selected_name = st.selectbox(
                "Model",
                options=model_options,
                index=default_index,
                help="Models and preview dates are managed in config/config_app.yaml",
            )
            selected_model = next(m for m in models if m["name"] == selected_name)
            st.caption(
                f"Deployment: `{selected_model['deployment']}` — "
                f"Preview: `{selected_model.get('preview', 'n/a')}` — "
                f"API version: `{selected_model.get('api_version', 'n/a')}`"
            )
            st.success("✅ Azure LLM ready — ambiguous pairs will be escalated")

            st.session_state["llm_config"] = {
                "enabled": True,
                "deployment": selected_model["deployment"],
                "api_version": selected_model.get("api_version", "2024-02-01"),
            }
    else:
        st.info("ℹ️ Running rule-based matching only — toggle on to enable Azure LLM")
        st.session_state["llm_config"] = {"enabled": False}

    # ── Proceed button ─────────────────────────────────────────────────────────
    st.markdown("---")
    ready = st.session_state["df1"] is not None and st.session_state["df2"] is not None

    if not ready:
        st.warning("Upload both DataFrames to continue.")
    else:
        if st.button("▶ Preview Assembled Names", type="primary", use_container_width=True):
            _assemble_and_preview()


def _assemble_and_preview():
    """Assemble names and navigate to preview page."""
    df1 = st.session_state["df1"]
    df2 = st.session_state["df2"]
    name_type = st.session_state["name_type"]
    df1_col_map = st.session_state["df1_col_map"]
    df2_col_map = st.session_state["df2_col_map"]

    # Validate
    errors1 = validate_column_map(df1, df1_col_map, name_type)
    errors2 = validate_column_map(df2, df2_col_map, name_type)

    if errors1 or errors2:
        for e in errors1 + errors2:
            st.error(e)
        return

    # Assemble
    df1_assembled = assemble_dataframe(df1, df1_col_map, name_type)
    df2_assembled = assemble_dataframe(df2, df2_col_map, name_type)

    st.session_state["df1_assembled"] = df1_assembled
    st.session_state["df2_assembled"] = df2_assembled
    st.session_state["page"] = "preview"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_preview():
    st.title("🔍 Preview Assembled Names")
    st.caption("Verify that names have been assembled correctly before running the match.")

    df1 = st.session_state["df1_assembled"]
    df2 = st.session_state["df2_assembled"]

    unique_a = extract_unique_names(df1)
    unique_b = extract_unique_names(df2)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### DataFrame 1")
        st.metric("Total rows", len(df1))
        st.metric("Unique names", len(unique_a))
        preview1 = pd.DataFrame({"Assembled Name": unique_a[:20]})
        st.dataframe(preview1, use_container_width=True, height=400)

    with col2:
        st.markdown("#### DataFrame 2")
        st.metric("Total rows", len(df2))
        st.metric("Unique names", len(unique_b))
        preview2 = pd.DataFrame({"Assembled Name": unique_b[:20]})
        st.dataframe(preview2, use_container_width=True, height=400)

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("◀ Back to Setup", use_container_width=True):
            st.session_state["page"] = "setup"
            st.rerun()
    with c3:
        if st.button("▶ Run Matching", type="primary", use_container_width=True):
            _run_matching()


def _run_matching():
    """Run the full matching pipeline."""
    df1 = st.session_state["df1_assembled"]
    df2 = st.session_state["df2_assembled"]
    name_type = st.session_state["name_type"]
    config = st.session_state["app_config"]
    llm_config = st.session_state["llm_config"]
    weights = get_weights(config, name_type)

    unique_a = extract_unique_names(df1)
    unique_b = extract_unique_names(df2)

    progress_bar = st.progress(0, text="Initializing...")

    def progress_callback(current, total):
        pct = current / total
        progress_bar.progress(pct, text=f"Matching {current}/{total} names...")

    start = time.time()
    match_results = match_unique_names(
        unique_a=unique_a,
        unique_b=unique_b,
        name_type=name_type,
        weights=weights,
        config=config,
        llm_config=llm_config if llm_config.get("enabled") else None,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - start

    progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")

    summary = compute_summary(match_results, len(df1), len(df2), unique_b)
    matched_df, only_df1_df, only_df2_df = build_output_dataframes(df1, df2, match_results)

    st.session_state["match_results"] = match_results
    st.session_state["summary"] = summary
    st.session_state["matched_df"] = matched_df
    st.session_state["only_df1_df"] = only_df1_df
    st.session_state["only_df2_df"] = only_df2_df
    st.session_state["page"] = "results"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_results():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        has_plotly = True
    except ImportError:
        has_plotly = False

    st.title("📊 Match Results")

    summary = st.session_state["summary"]
    matched_df = st.session_state["matched_df"]
    only_df1_df = st.session_state["only_df1_df"]
    only_df2_df = st.session_state["only_df2_df"]

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("DF1 Total", summary["df1_total"])
    m2.metric("DF2 Total", summary["df2_total"])
    m3.metric("✅ Matched Pairs", summary["matched_pairs"])
    m4.metric("🔵 Only in DF1", summary["only_in_a"])
    m5.metric("🟡 Only in DF2", summary["only_in_b"])

    st.markdown("---")

    # ── Plotly charts ──────────────────────────────────────────────────────────
    if has_plotly:
        st.subheader("Visualizations")
        ch1, ch2, ch3, ch4 = st.columns(4)

        cat_counts = summary["category_counts"]

        # Chart 1 — Donut
        with ch1:
            fig = go.Figure(go.Pie(
                labels=["Matched", "Only DF1", "Only DF2"],
                values=[
                    summary["matched_names_a"],
                    summary["only_in_a"],
                    summary["only_in_b"],
                ],
                hole=0.5,
                marker_colors=["#22c55e", "#3b82f6", "#f59e0b"],
            ))
            fig.update_layout(title="Overall Distribution", height=280, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        # Chart 2 — Category bar
        with ch2:
            fig = go.Figure(go.Bar(
                x=list(cat_counts.values()),
                y=list(cat_counts.keys()),
                orientation="h",
                marker_color=["#22c55e", "#84cc16", "#f59e0b", "#ef4444"],
            ))
            fig.update_layout(title="Match Categories", height=280, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        # Chart 3 — Score histogram
        with ch3:
            if not matched_df.empty and "Score" in matched_df.columns:
                fig = px.histogram(
                    matched_df, x="Score", nbins=20,
                    title="Score Distribution",
                    color_discrete_sequence=["#6366f1"],
                )
                fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        # Chart 4 — Funnel
        with ch4:
            total_possible = summary["unique_a"] * summary["unique_b"]
            fig = go.Figure(go.Funnel(
                y=["Unique pairs possible", "After blocking", "LLM escalated", "Final matched"],
                x=[
                    total_possible,
                    max(summary["matched_pairs"] * 3, summary["matched_pairs"]),
                    summary["llm_escalations"],
                    summary["matched_pairs"],
                ],
                marker_color=["#6366f1", "#8b5cf6", "#a78bfa", "#22c55e"],
            ))
            fig.update_layout(title="Matching Funnel", height=280, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Avg match score: **{summary['avg_score']}** | LLM escalations: **{summary['llm_escalations']}**")
    else:
        st.info("Install plotly to see charts: pip install plotly")

    st.markdown("---")

    # ── Results tabs ───────────────────────────────────────────────────────────
    st.subheader("Results")
    tab1, tab2, tab3 = st.tabs([
        f"✅ Matched ({len(matched_df)})",
        f"🔵 Only in DF1 ({len(only_df1_df)})",
        f"🟡 Only in DF2 ({len(only_df2_df)})",
    ])

    with tab1:
        if not matched_df.empty:
            search = st.text_input("Search matched results", key="search_matched", placeholder="Filter by any value...")
            df_show = matched_df
            if search:
                mask = df_show.astype(str).apply(lambda col: col.str.contains(search, case=False)).any(axis=1)
                df_show = df_show[mask]
            st.dataframe(df_show, use_container_width=True, height=400)
        else:
            st.info("No matches found above threshold.")

    with tab2:
        if not only_df1_df.empty:
            st.dataframe(only_df1_df, use_container_width=True, height=400)
        else:
            st.success("All DF1 names found a match in DF2.")

    with tab3:
        if not only_df2_df.empty:
            st.dataframe(only_df2_df, use_container_width=True, height=400)
        else:
            st.success("All DF2 names were matched by at least one DF1 name.")

    # ── Download section ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Download Results")

    excel_bytes = export_to_excel(matched_df, only_df1_df, only_df2_df)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            label="⬇ Download Excel (all 3 sheets)",
            data=excel_bytes,
            file_name="name_match_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d2:
        if not matched_df.empty:
            st.download_button(
                label="⬇ Matched CSV",
                data=export_csv(matched_df),
                file_name="matched.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with d3:
        if not only_df1_df.empty:
            st.download_button(
                label="⬇ Only DF1 CSV",
                data=export_csv(only_df1_df),
                file_name="only_df1.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with d4:
        if not only_df2_df.empty:
            st.download_button(
                label="⬇ Only DF2 CSV",
                data=export_csv(only_df2_df),
                file_name="only_df2.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    if st.button("◀ Start Over", use_container_width=False):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Page indicator
    pages = {"setup": "1", "preview": "2", "results": "3"}
    current = st.session_state["page"]

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown(f"**Step {pages.get(current, '1')} of 3**")
    for p, num in pages.items():
        label = {"setup": "⚙️ Setup", "preview": "🔍 Preview", "results": "📊 Results"}[p]
        if p == current:
            st.sidebar.markdown(f"**→ {label}**")
        else:
            st.sidebar.markdown(f"&nbsp;&nbsp; {label}")

    if current == "setup":
        page_setup()
    elif current == "preview":
        page_preview()
    elif current == "results":
        page_results()


if __name__ == "__main__":
    main()