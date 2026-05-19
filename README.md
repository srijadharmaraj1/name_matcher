# Name Matching Tool

Record linkage across two name lists — persons or entities — with fuzzy, phonetic, structural, and Azure LLM-powered matching.

---

## Project Structure

```
name-matcher/
├── main.py                        CLI entry point
├── app.py                         Streamlit UI entry point
├── config/
│   ├── config_app.yaml            Shared — thresholds, weights, model list
│   ├── config_cli.yaml            CLI — paths, columns, type, azure on/off
│   ├── stopwords.yaml             Entity stop words (Ltd, Inc, Corp ...)
│   ├── particles.yaml             Person particles + titles + suffixes
│   └── .env.example               Copy to .env and fill credentials
├── matcher/
│   ├── normalizer.py              Unicode, diacritics, lowercase, punctuation
│   ├── assembler.py               Column → assembled full name string
│   ├── classifier.py              Person vs Entity auto-detection
│   ├── phonetic.py                Soundex + Double Metaphone + Caverphone + Jaro-Winkler
│   ├── fuzzy.py                   Edit distance + Token Sort + Token Set Ratio
│   ├── entity.py                  Stop word stripping, core token matching
│   ├── structural.py              Initials, missing middle, token subset
│   ├── scorer.py                  Weighted signal combiner → score + category + reasons
│   ├── llm.py                     Azure OpenAI via Service Principal + proxy
│   └── pipeline.py                Orchestrates full match flow + blocking
├── utils/
│   ├── config_loader.py           YAML loader + validation helpers
│   ├── deduper.py                 Extract unique names for efficient matching
│   ├── merger.py                  Merge results back to full dataframes
│   └── exporter.py                3-sheet Excel + CSV export
└── tests/
    ├── test_normalizer.py
    ├── test_scorer.py
    └── fixtures/
        ├── sample_df1.csv
        └── sample_df2.csv
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp config/.env.example config/.env
```

Edit `config/.env`:

```env
# Azure Service Principal
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Corporate Proxy (Windows auth)
HTTPS_PROXY_HOST=proxyhost.company.com
HTTPS_PROXY_PORT=8080
WINDOWS_USERNAME=DOMAIN\\your-username
WINDOWS_PASSWORD=your-windows-password
```

---

## Running

### Streamlit UI

```bash
streamlit run app.py
```

**Flow:**
1. Select type — Person or Entity
2. Upload DF1 → map name columns
3. Upload DF2 → map name columns
4. Toggle Azure LLM on/off → select model if on
5. Preview assembled names
6. Run matching
7. View results + download Excel (3 sheets)

---

### CLI

Edit `config/config_cli.yaml`:

```yaml
data:
  df1_path: data/list1.csv
  df2_path: data/list2.csv
  output_path: output/results.xlsx

name_type: person   # person or entity

df1_columns:
  first: first_name
  middle: middle_name
  last: last_name
  full: null        # set this to override split columns

df2_columns:
  full: name        # single column

azure:
  enabled: true     # true = use LLM for ambiguous band, false = rule-based only
  model: gpt-4o     # must match a name in config_app.yaml azure.models
```

Then run:

```bash
python main.py
```

---

## Output — 3 Sheets

| Sheet | Contents |
|---|---|
| **Matched** | DF1 rows + DF2 matched rows + Score + Category + Reason + Match_Number |
| **Only in DF1** | DF1 rows with no match found in DF2 |
| **Only in DF2** | DF2 rows never matched by any DF1 name |

One row per matched pair (vertical). One-to-many supported — if a name matches 3 names in DF2, it appears as 3 rows.

---

## Matching Pipeline

```
Assemble name from columns
        ↓
Normalize — unicode, diacritics, lowercase, punctuation, hyphens
        ↓
Person → sort tokens alphabetically (order-agnostic)
Entity → strip stop words for matching, preserve order for display
        ↓
Blocking keys — first 3 chars + soundex per token (reduces candidate pairs)
        ↓
Multi-signal scoring:
  • Exact token set match
  • Fuzzy — edit distance + token sort + token set ratio (max of 3)
  • Phonetic — soundex + double metaphone + caverphone (voting)
  • Structural — initials (capped), missing middle, token subset
  • Jaro-Winkler — prefix-sensitive similarity
        ↓
Weighted score (person weights ≠ entity weights — see config_app.yaml)
        ↓
Score > 90  → Exact match — done
Score < 35  → No Match — done
Score 35–90 → Azure LLM escalation (if enabled)
              LLM returns adjusted score + confidence + plain English reason
        ↓
Deduplicate → match → merge back to full dataframes → 3-sheet Excel
```

---

## Configuration Reference

### config_app.yaml — shared settings

| Key | Description |
|---|---|
| `thresholds.exact` | Score ≥ this → Exact category (default 95) |
| `thresholds.llm_band_lower` | Score above this → LLM escalation (default 35) |
| `thresholds.llm_band_upper` | Score below this → LLM escalation (default 90) |
| `person_weights` | Signal weights for person matching (must sum to 1.0) |
| `entity_weights` | Signal weights for entity matching (must sum to 1.0) |
| `matching.max_matches_per_name` | Max DF2 matches per DF1 name (default 5) |
| `matching.initial_match_max_score` | Cap on initial-only matches (default 70) |
| `azure.models` | List of deployments — name, deployment, api_version, preview, default |

### config_cli.yaml — CLI settings

| Key | Description |
|---|---|
| `data.df1_path` | Path to first CSV/Excel file |
| `data.df2_path` | Path to second CSV/Excel file |
| `data.output_path` | Output Excel path |
| `name_type` | `person` or `entity` |
| `df1_columns` / `df2_columns` | Column mapping (full overrides split columns) |
| `azure.enabled` | `true` to use LLM, `false` for rule-based only |
| `azure.model` | Model name — must match an entry in config_app.yaml |

---

## Adding / Updating Azure Models

Edit the `azure.models` list in `config_app.yaml`:

```yaml
azure:
  models:
    - name: gpt-4o
      deployment: gpt-4o
      api_version: "2024-05-01-preview"
      preview: "2025-05"
      default: true
    - name: gpt-4o-mini
      deployment: gpt-4o-mini
      api_version: "2024-05-01-preview"
      preview: "2025-05"
      default: false
```

- Update `preview` when Microsoft releases new preview versions
- Update `api_version` to match your Azure deployment
- Set `default: true` on whichever model should auto-select in the UI

---

## Running Tests

```bash
python tests/test_normalizer.py
python tests/test_scorer.py
```

Install `jellyfish` for stronger phonetic matching:
```bash
pip install jellyfish
```
Without it the tool falls back to built-in Soundex and simplified Metaphone — still functional, slightly lower phonetic recall.
