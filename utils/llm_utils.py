"""
LLM utilities for ICF linking.

Supports two providers:
  - ollama   : local/server models via Ollama /api/chat (default)
  - openai   : GPT-4o, o3, o4-mini via OpenAI API

Provider is selected automatically by model name, or can be passed explicitly.
Credentials loaded from .env:
  Ollama    : OLLAMA_API_URL, COSYBIO_API_KEY, LLM_MODEL
  OpenAI    : OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import urllib3
import requests
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

OLLAMA_API_URL  = os.environ["OLLAMA_API_URL"].rstrip("/")
API_KEY         = os.environ["COSYBIO_API_KEY"]
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3.6:35b")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
TIMEOUT         = 300  # seconds per request (70B models need up to 5 min)

# Optional extra providers from an untracked local module (utils/local_providers.py)
try:
    from utils.local_providers import PROVIDERS as _EXTRA_PROVIDERS
except ImportError:
    _EXTRA_PROVIDERS = {}


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    for name, p in _EXTRA_PROVIDERS.items():
        if p["match"](model):
            return name
    m = model.lower()
    # gpt-oss is a local Ollama model despite the gpt- prefix
    if m.startswith("gpt-oss"):
        return "ollama"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    return "ollama"


_SYSTEM_BASE = """You are an expert clinical coder specialising in the ICF \
(International Classification of Functioning, Disability and Health, German version BfArM 2005).

Your task is to link German patient goal concepts ("Konzepte") to the most appropriate \
ICF code(s) following the Cieza et al. 2005 linking rules:

1. Focus on the explicit meaning — do not infer what is implied.
2. Link to the most specific code possible (prefer 4th or 5th level over chapter level).
3. A single concept may need 1–4 codes if it covers distinct ICF aspects.
4. Use non-standard codes only when no standard code fits:
   • nd  = not definable (meaning too vague for ICF)
   • nc  = not covered by ICF
   • pf  = personal factor
   • hc  = health condition (diagnosis, not functioning)
5. Environmental factors (e-codes): external facilitators or barriers.
6. Body structures (s-codes): anatomy. Body functions (b-codes): physiology.
7. Activities & participation (d-codes): what a person does or can do.

Respond ONLY with a JSON object — no prose outside the JSON."""


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _parse_json_from_text(text: str) -> dict:
    """Extract the JSON object containing 'predicted_codes' from model output."""
    # Strip known thinking tag patterns (Qwen3, phi4, deepseek, etc.)
    for pattern in [r"<think>.*?</think>", r"<\|think\|>.*?<\|/think\|>"]:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = text.strip()

    # Strategy 1: locate 'predicted_codes' key, walk outward to find the JSON block.
    # More robust than greedy \{.*\} when reasoning text contains braces.
    idx = text.rfind('"predicted_codes"')
    if idx != -1:
        start = text.rfind("{", 0, idx)
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:]):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : start + i + 1])
                        except json.JSONDecodeError:
                            break

    # Strategy 2: try each flat {...} block from last to first
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict) and "predicted_codes" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Strategy 3: greedy fallback (original behaviour)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {"predicted_codes": [], "reasoning": text[:300], "parse_error": True}


def _call_openai(user_message: str, system: str, model: str, retries: int = 3) -> dict:
    """Call OpenAI API (GPT-4o, o3, o4-mini)."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"predicted_codes": [], "reasoning": "openai package not installed", "error": True}

    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_message},
                ],
                timeout=TIMEOUT,
            )
            content = resp.choices[0].message.content
            return _parse_json_from_text(content)
        except Exception as e:
            if attempt == retries - 1:
                return {"predicted_codes": [], "reasoning": str(e), "error": True}
            time.sleep(5 * (attempt + 1))
    return {"predicted_codes": [], "reasoning": "max retries exceeded", "error": True}


def call_llm(
    user_message: str,
    system_suffix: str = "",
    retries: int = 3,
    model: Optional[str] = None,
    think: bool = True,
    provider: Optional[str] = None,
) -> dict:
    """
    Send one request to Ollama (default) or OpenAI.
    Provider is auto-detected from model name, or pass provider= explicitly.
    model: override LLM_MODEL from .env.
    think: set False for models without reasoning capability.
    Returns dict with 'predicted_codes' (list[str]) and 'reasoning' (str).
    """
    system = _SYSTEM_BASE
    if system_suffix:
        system = system + "\n\n" + system_suffix

    _model = model or LLM_MODEL
    _provider = provider or _detect_provider(_model)

    # Route to the correct provider
    if _provider == "openai":
        return _call_openai(user_message, system, _model, retries)
    if _provider in _EXTRA_PROVIDERS:
        return _EXTRA_PROVIDERS[_provider]["call"](user_message, system, _model, think, retries)

    # Default: Ollama (streaming to survive proxy timeouts)
    payload = {
        "model": _model,
        "stream": True,
        "think": think,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_API_URL}/api/chat",
                headers=_headers(),
                json=payload,
                stream=True,
                timeout=TIMEOUT,
                verify=False,
            )
            resp.raise_for_status()

            # Accumulate streamed content
            content = ""
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})
                    content += msg.get("content", "")
                    if chunk.get("done"):
                        break

            return _parse_json_from_text(content)

        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            print(f"  Timeout on attempt {attempt+1}, waiting {wait}s...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if attempt == retries - 1:
                raise
            print(f"  HTTP error ({e}), retrying...")
            time.sleep(5)
        except (ValueError, KeyError, requests.exceptions.JSONDecodeError) as e:
            wait = 10 * (attempt + 1)
            print(f"  Bad response on attempt {attempt+1} ({e}), waiting {wait}s...")
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 10 * (attempt + 1)
            print(f"  Connection error on attempt {attempt+1}, waiting {wait}s...")
            time.sleep(wait)

    return {"predicted_codes": [], "reasoning": "max retries exceeded", "error": True}


# ─── Prompt builders ────────────────────────────────────────────────────────

def build_zeroshot_prompt(konzept: str, icf_list_text: str) -> str:
    return f"""Link this German patient goal concept to ICF code(s).

Concept: "{konzept}"

Available ICF codes (code - German title):
{icf_list_text}

Return 1–4 ICF codes that best match this concept, ordered by relevance.

Respond with this exact JSON format:
{{"predicted_codes": ["code1", ...], "reasoning": "brief explanation"}}"""


def build_fewshot_prompt(
    konzept: str,
    icf_list_text: str,
    examples: list[dict],
) -> str:
    ex_lines = []
    for i, ex in enumerate(examples, 1):
        codes_str = ", ".join(ex["gold_codes"])
        ex_lines.append(
            f"Example {i}:\n  Concept: \"{ex['konzept']}\"\n  ICF codes: {codes_str}"
        )
    examples_block = "\n\n".join(ex_lines)

    return f"""Link this German patient goal concept to ICF code(s).

--- Reference examples (expert-annotated) ---
{examples_block}
--- End of examples ---

Available ICF codes (code - German title):
{icf_list_text}

Now link this concept:
Concept: "{konzept}"

Return 1–4 ICF codes that best match this concept, ordered by relevance.

Respond with this exact JSON format:
{{"predicted_codes": ["code1", ...], "reasoning": "brief explanation"}}"""


def build_rag_prompt(
    konzept: str,
    candidate_codes: list[dict],
    similar_examples: list[dict],
) -> str:
    cand_lines = [
        f"  {c['code']} - {c['title_de']} (similarity: {c['similarity']:.3f})"
        for c in candidate_codes
    ]
    cand_block = "\n".join(cand_lines)

    ex_lines = []
    for i, ex in enumerate(similar_examples, 1):
        codes_str = ", ".join(ex["gold_codes"])
        ex_lines.append(
            f"Example {i} (similarity: {ex['similarity']:.3f}):\n"
            f"  Concept: \"{ex['konzept']}\"\n"
            f"  ICF codes: {codes_str}"
        )
    ex_block = "\n\n".join(ex_lines)

    return f"""Link this German patient goal concept to ICF code(s).

--- Similar expert-annotated concepts (retrieved by semantic similarity) ---
{ex_block}
--- End of examples ---

--- Candidate ICF codes (retrieved by semantic similarity to the concept) ---
{cand_block}
--- End of candidates ---

Note: candidates are ranked by similarity but may not all be correct.
Use your ICF expertise and the examples to choose the most appropriate code(s).

Concept: "{konzept}"

Return 1–4 ICF codes that best match this concept, ordered by relevance.

Respond with this exact JSON format:
{{"predicted_codes": ["code1", ...], "reasoning": "brief explanation"}}"""
