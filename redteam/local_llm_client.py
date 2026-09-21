"""
Optional client for a local, OpenAI-compatible model server — e.g.
Ollama, vLLM, LM Studio, text-generation-webui. Lets the LLM-generation
path work without an Anthropic API key if you'd rather run everything
against a local model.

Configure via environment variables:
  LOCAL_MODEL_ENDPOINT   e.g. "http://localhost:11434" (Ollama) or
                          "http://localhost:8000" (vLLM/LM Studio)
  LOCAL_MODEL_NAME        e.g. "llama3.1" / "mistral" / whatever you've pulled

If LOCAL_MODEL_ENDPOINT isn't set, `is_configured()` returns False and
the caller should fall back to an empty batch (see generate_three_batches.py).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import urllib.request
import urllib.error

from .schema import Attack, AttackCategory, AttackSource
from .llm_client import _SYSTEM_PROMPT, AttackGeneratorLLM


def is_configured() -> bool:
    return bool(os.environ.get("LOCAL_MODEL_ENDPOINT"))


class LocalModelAttackGenerator:
    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None):
        self.endpoint = (endpoint or os.environ.get("LOCAL_MODEL_ENDPOINT", "")).rstrip("/")
        self.model = model or os.environ.get("LOCAL_MODEL_NAME", "llama3.1")

    def _call_model(self, user_content: str) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1500,
            "temperature": 0.9,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"Local model at {url} unreachable: {e}") from e

        return data["choices"][0]["message"]["content"]

    def generate_variants(
        self,
        category: AttackCategory,
        example_templates: List[str],
        n: int = 5,
        inspiration: Optional[List[str]] = None,
    ) -> List[Attack]:
        prompt_parts = [
            f"Category: {category.value}",
            f"Generate {n} new test cases.",
            "Example templates already known (do not just repeat these):",
        ]
        prompt_parts.extend(f"- {t}" for t in example_templates[:5])
        if inspiration:
            prompt_parts.append(
                "\nThese recent variants evaded the defender in the last "
                "round — generate stronger cases building on what worked:"
            )
            prompt_parts.extend(f"- {t}" for t in inspiration[:5])

        raw = self._call_model("\n".join(prompt_parts))
        # Reuse the same forgiving JSON parser as the Anthropic client.
        return AttackGeneratorLLM._parse_response(raw, category)
