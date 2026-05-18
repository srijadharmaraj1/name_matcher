"""
llm.py — OpenRouter integration for ambiguous name pair resolution.
Only called for pairs in the 35–90 score band.
Returns adjusted score, confidence, and human-readable reason.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "config" / ".env")


def _get_client(api_key: str = None, base_url: str = None):
    """Initialize OpenRouter client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    base_url = base_url or os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    if not api_key:
        raise ValueError("OpenRouter API key must be provided.")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def _build_prompt(
    name_a: str,
    name_b: str,
    name_type: str,
    signals: dict,
    rule_score: float,
) -> str:
    signal_lines = "\n".join(
        f"  - {k}: {v*100:.1f}%"
        for k, v in signals.items()
        if isinstance(v, float)
    )

    return f"""You are an expert name matching system.

Analyze whether these two {name_type} names refer to the same {name_type}.

Name A: "{name_a}"
Name B: "{name_b}"
Type: {name_type}

Rule-based signals already computed:
{signal_lines}

Rule-based score: {rule_score:.1f}/100

Consider:
- Spelling variations
- Phonetic similarities
- Cultural name variants
- Initials vs full names
- Transliteration differences
- Company legal suffixes (Ltd, Inc, LLC, Pvt, etc.)

Be conservative with ambiguous matches.

Respond ONLY with valid JSON:

{{
  "is_match": true,
  "confidence": 0-100,
  "adjusted_score": 0-100,
  "reason": "short explanation"
}}
"""


def call_llm(
    name_a: str,
    name_b: str,
    name_type: str,
    signals: dict,
    rule_score: float,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    max_tokens: int = 300,
    temperature: float = 0,
) -> dict:
    """
    Call OpenRouter LLM to resolve an ambiguous name pair.

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

    model = model or os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini",
    )

    try:
        client = _get_client(api_key, base_url)

        prompt = _build_prompt(
            name_a,
            name_b,
            name_type,
            signals,
            rule_score,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON API. "
                        "Always return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if content.startswith("```"):
            content = content.split("```")[1]

            if content.startswith("json"):
                content = content[4:]

        content = content.strip()

        result = json.loads(content)

        return {
            "is_match": bool(result.get("is_match", False)),
            "confidence": int(result.get("confidence", 50)),
            "adjusted_score": int(
                result.get("adjusted_score", rule_score)
            ),
            "reason": result.get("reason", "LLM assessment"),
            "llm_used": True,
        }

    except Exception as e:
        return {
            "is_match": rule_score >= 50,
            "confidence": 50,
            "adjusted_score": int(rule_score),
            "reason": (
                f"LLM unavailable ({str(e)[:80]}); "
                "rule-based score used"
            ),
            "llm_used": False,
        }


def is_llm_configured(api_key: str = None) -> bool:
    """Check if OpenRouter credentials are available."""
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    return bool(key)