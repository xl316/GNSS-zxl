from __future__ import annotations

import os
from typing import Optional


class LLMClient:
    """Optional LLM adapter.

    The rest of the project works without an API key. If OPENAI_API_KEY is set,
    this client uses the OpenAI Responses API for writing-style review and report polishing.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception:
                self.enabled = False
                self._client = None

    def complete(self, system: str, user: str, max_chars: int = 6000) -> str:
        if not self.enabled or self._client is None:
            return ""
        user = user[:max_chars]
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
            )
            return getattr(response, "output_text", "") or ""
        except Exception as exc:
            return f"[LLM 调用失败：{exc}]"
