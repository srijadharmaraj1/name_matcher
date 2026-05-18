# Name Matching Tool

A Python project for matching names across two datasets using a combination of fuzzy, phonetic, structural, and optional AI-powered matching.

## Features

- Match person names or entity/company names
- Upload CSV or Excel files via Streamlit UI
- CLI mode for batch matching and Excel exports
- Configurable thresholds, weights, and matching settings
- Optional Azure OpenAI LLM support for ambiguous match review

## Requirements

- Python 3.11+ recommended
- `requirements.txt` contains the main dependencies

## Installation

1. Clone or open the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

```powershell
.\.venv\Scripts\Activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Streamlit UI

Run the interactive app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

### Workflow

1. Choose name type: `person` or `entity`
2. Upload two datasets (`CSV` or `Excel`)
3. Map the relevant name columns for each dataset
4. Adjust matching thresholds and settings
5. View and export matched results

## CLI Usage

Run the matching pipeline from the terminal:

```bash
python main.py
```

By default the CLI uses `config/config_cli.yaml`. To specify a different config file:

```bash
python main.py --config config/config_cli.yaml
```

## Configuration

- `config/config_app.yaml` contains shared matching settings, thresholds, and weight values.
- `config/config_cli.yaml` contains CLI data paths and column mappings.
- `config/stopwords.yaml` contains entity stopwords used for company/entity matching.

### CLI config example

Update `df1_path` and `df2_path` to point to your input files, and set the mapping keys for each name column.

For `entity` mode, only `full` is required:

```yaml
name_type: entity

df1_columns:
  full: company_name

df2_columns:
  full: entity_name
```

## Environment Variables

The app and CLI load `.env` values from `config/.env` for Azure OpenAI settings.

Common variables include:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`

If Azure LLM is not configured, the tool falls back to rule-based matching.

## Project Structure

- `app.py` — Streamlit application entry point
- `main.py` — CLI entry point
- `matcher/` — Matching algorithms and pipeline
- `utils/` — config loading, deduping, exporting, and merging helpers
- `config/` — configuration files and stopwords

## Notes

- Input files may be `.csv`, `.xls`, or `.xlsx`
- The tool can export results to Excel via the CLI
- The Streamlit UI supports interactive mapping and preview before export
