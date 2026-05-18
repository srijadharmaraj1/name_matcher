"""
config_loader.py — Load and validate YAML config files.
Merges CLI config and app config cleanly.
"""

from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_app_config() -> dict:
    """Load shared app config (thresholds, weights, matching params)."""
    return load_yaml("config_app.yaml")


def load_cli_config() -> dict:
    """Load CLI-specific config (paths, type, column names)."""
    return load_yaml("config_cli.yaml")


def get_weights(config: dict, name_type: str) -> dict:
    """Extract the correct weight set for name type."""
    key = f"{name_type}_weights"
    return config.get(key, config.get("person_weights", {}))


def validate_cli_config(cli_config: dict, df1: object, df2: object) -> list[str]:
    """
    Validate CLI config against loaded dataframes.
    Returns list of error strings (empty = valid).
    """
    errors = []
    name_type = cli_config.get("name_type", "")
    if name_type not in ("person", "entity"):
        errors.append(f"name_type must be 'person' or 'entity', got: '{name_type}'")

    for df_key, df in [("df1_columns", df1), ("df2_columns", df2)]:
        col_map = cli_config.get(df_key, {})
        if name_type == "entity":
            col = col_map.get("full")
            if not col:
                errors.append(f"{df_key}: entity type requires 'full' column")
            elif col not in df.columns:
                errors.append(f"{df_key}: column '{col}' not found")
        else:
            full = col_map.get("full")
            first = col_map.get("first")
            last = col_map.get("last")
            if full:
                if full not in df.columns:
                    errors.append(f"{df_key}: column '{full}' not found")
            elif not (first and last):
                errors.append(f"{df_key}: person type requires 'full' or both 'first'+'last'")
            else:
                for field in ["first", "last", "middle", "title", "suffix"]:
                    col = col_map.get(field)
                    if col and col not in df.columns:
                        errors.append(f"{df_key}: column '{col}' not found")

    return errors
