"""
llm.py — Azure OpenAI integration for ambiguous name pair resolution.
Only called for pairs in the 35–90 score band.
Returns adjusted score, confidence, and human-readable reason.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / ".env")


def _get_client(endpoint: str = None, api_key: str = None, api_version: str = None):
    """Initialize Azure OpenAI client."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not endpoint or not api_key:
        raise ValueError("Azure OpenAI endpoint and API key must be provided.")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _build_prompt(name_a: str, name_b: str, name_type: str, signals: dict, rule_score: float) -> str:
    signal_lines = "\n".join(
        f"  - {k}: {v*100:.1f}%" for k, v in signals.items() if isinstance(v, float)
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


def call_llm(
    name_a: str,
    name_b: str,
    name_type: str,
    signals: dict,
    rule_score: float,
    deployment: str = None,
    endpoint: str = None,
    api_key: str = None,
    api_version: str = None,
    max_tokens: int = 500,
    temperature: float = 0,
) -> dict:
    """
    Call Azure OpenAI to resolve an ambiguous name pair.

    Returns:
        {
            is_match: bool,
            confidence: int,
            adjusted_score: int,
            reason: str,
            llm_used: True
        }
    On failure, returns original rule-based score with error reason.
    """
    deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    try:
        client = _get_client(endpoint, api_key, api_version)
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
            "is_match": result.get("is_match", False),
            "confidence": int(result.get("confidence", 50)),
            "adjusted_score": int(result.get("adjusted_score", rule_score)),
            "reason": result.get("reason", "LLM assessment"),
            "llm_used": True,
        }

    except Exception as e:
        # Graceful fallback — return rule-based score, flag error
        return {
            "is_match": rule_score >= 50,
            "confidence": 50,
            "adjusted_score": int(rule_score),
            "reason": f"LLM unavailable ({str(e)[:80]}); rule-based score used",
            "llm_used": False,
        }


def is_llm_configured(endpoint: str = None, api_key: str = None) -> bool:
    """Check if Azure credentials are available."""
    ep = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
    return bool(ep and key)
