"""
llm.py — Azure OpenAI integration using Service Principal authentication.
Auth flow:
  1. ClientSecretCredential (tenant_id + client_id + client_secret) → Bearer token
  2. Token + proxy → Azure OpenAI endpoint call
  3. Proxy constructed from HTTPS_PROXY_HOST, HTTPS_PROXY_PORT, WINDOWS_USERNAME, WINDOWS_PASSWORD

Only called for name pairs in the ambiguous score band (default 35-90).
Returns adjusted score, confidence, and human-readable reason.
"""

import json
import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / ".env")


# ── PROXY ─────────────────────────────────────────────────────────────────────

def _build_proxy_url() -> str | None:
    """
    Construct proxy URL from individual .env components.
    Format: http://DOMAIN\\username:password@host:port
    Returns None if host not configured.
    """
    host = os.getenv("HTTPS_PROXY_HOST", "").strip()
    port = os.getenv("HTTPS_PROXY_PORT", "").strip()
    username = os.getenv("WINDOWS_USERNAME", "").strip()
    password = os.getenv("WINDOWS_PASSWORD", "").strip()

    if not host:
        return None

    if username and password:
        # URL-encode to safely handle backslashes in DOMAIN\\user and special chars
        encoded_user = quote(username, safe="")
        encoded_pass = quote(password, safe="")
        proxy = f"http://{encoded_user}:{encoded_pass}@{host}"
    else:
        proxy = f"http://{host}"

    if port:
        proxy = f"{proxy}:{port}"

    return proxy


# ── TOKEN ─────────────────────────────────────────────────────────────────────

def _get_bearer_token(proxy_url: str | None = None) -> str:
    """
    Obtain Bearer token from Azure AD using ClientSecretCredential.
    Injects proxy into the credential transport if proxy is configured.
    """
    try:
        from azure.identity import ClientSecretCredential
    except ImportError:
        raise ImportError(
            "azure-identity not installed. Run: pip install azure-identity"
        )

    tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
    client_id = os.getenv("AZURE_CLIENT_ID", "").strip()
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "").strip()

    if not all([tenant_id, client_id, client_secret]):
        raise ValueError(
            "Missing Azure credentials. Ensure AZURE_TENANT_ID, AZURE_CLIENT_ID, "
            "and AZURE_CLIENT_SECRET are set in config/.env"
        )

    if proxy_url:
        try:
            from azure.core.pipeline.transport import RequestsTransport
            transport = RequestsTransport(
                proxies={"https": proxy_url, "http": proxy_url}
            )
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                transport=transport,
            )
        except Exception:
            # Fallback without custom transport if injection fails
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
    else:
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    scope = "https://cognitiveservices.azure.com/.default"
    token = credential.get_token(scope)
    return token.token


# ── CLIENT ────────────────────────────────────────────────────────────────────

def _get_openai_client(proxy_url: str | None, api_version: str):
    """
    Build AzureOpenAI client authenticated via Bearer token.
    Passes proxy via httpx if configured.
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not set in config/.env")

    bearer_token = _get_bearer_token(proxy_url)

    if proxy_url:
        try:
            import httpx
            http_client = httpx.Client(
                proxies={"https://": proxy_url, "http://": proxy_url},
                verify=True,
            )
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_version=api_version,
                azure_ad_token=bearer_token,
                http_client=http_client,
            )
        except ImportError:
            pass  # httpx not available, fall through

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token=bearer_token,
    )


# ── PROMPT ────────────────────────────────────────────────────────────────────

def _build_prompt(
    name_a: str,
    name_b: str,
    name_type: str,
    signals: dict,
    rule_score: float,
) -> str:
    signal_lines = "\n".join(
        f"  - {k}: {v * 100:.1f}%"
        for k, v in signals.items()
        if isinstance(v, float)
    )
    return f"""You are an expert name matching system. Analyze whether these two {name_type} names refer to the same {name_type}.

Name A: "{name_a}"
Name B: "{name_b}"
Type: {name_type}

Rule-based signals already computed:
{signal_lines}
Rule-based score: {rule_score:.1f}/100

Consider:
- Spelling variations, phonetic similarities, cultural name variants
- Initials vs full names (be conservative — J. Smith could be many people)
- For entities: core name similarity ignoring legal suffixes (Ltd, Inc, etc.)
- For persons: same person could have different middle name usage, name order, transliteration

Respond ONLY with a valid JSON object, no preamble, no markdown:
{{
  "is_match": true or false,
  "confidence": <integer 0-100>,
  "adjusted_score": <integer 0-100>,
  "reason": "<one clear sentence explaining your decision>"
}}"""


# ── MAIN CALL ─────────────────────────────────────────────────────────────────

def call_llm(
    name_a: str,
    name_b: str,
    name_type: str,
    signals: dict,
    rule_score: float,
    deployment: str = None,
    api_version: str = None,
    max_tokens: int = 500,
    temperature: float = 0,
) -> dict:
    """
    Call Azure OpenAI to resolve an ambiguous name pair.
    Uses service principal auth + corporate proxy from .env.

    Returns:
        {
            is_match: bool,
            confidence: int,
            adjusted_score: int,
            reason: str,
            llm_used: bool
        }
    On any failure returns rule-based score with error reason — never raises.
    """
    api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    try:
        proxy_url = _build_proxy_url()
        client = _get_openai_client(proxy_url, api_version)
        prompt = _build_prompt(name_a, name_b, name_type, signals, rule_score)

        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)

        return {
            "is_match": bool(result.get("is_match", False)),
            "confidence": int(result.get("confidence", 50)),
            "adjusted_score": int(result.get("adjusted_score", rule_score)),
            "reason": result.get("reason", "LLM assessment"),
            "llm_used": True,
        }

    except Exception as e:
        return {
            "is_match": rule_score >= 50,
            "confidence": 50,
            "adjusted_score": int(rule_score),
            "reason": f"LLM unavailable ({str(e)[:100]}); rule-based score used",
            "llm_used": False,
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def is_llm_configured() -> bool:
    """
    Check minimum Azure Service Principal credentials are present.
    Does not validate them — just checks non-empty.
    """
    return all([
        os.getenv("AZURE_TENANT_ID", "").strip(),
        os.getenv("AZURE_CLIENT_ID", "").strip(),
        os.getenv("AZURE_CLIENT_SECRET", "").strip(),
        os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
    ])


def get_models_from_config(app_config: dict) -> tuple[list[dict], dict]:
    """
    Extract model list and default model from app config.

    Returns:
        (models_list, default_model)
        models_list: full list of model dicts from config_app.yaml
        default_model: entry with default: true, or first entry if none marked
    """
    models = app_config.get("azure", {}).get("models", [])
    if not models:
        fallback = {
            "name": "gpt-4o",
            "deployment": "gpt-4o",
            "api_version": "2024-02-01",
            "preview": "",
            "default": True,
        }
        return [fallback], fallback

    default = next((m for m in models if m.get("default")), models[0])
    return models, default