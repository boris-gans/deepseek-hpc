"""Client wrapper around the Fireworks AI chat completion endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import requests


logger = logging.getLogger(__name__)


class FireworksAICompletionClient:
    """Handles prompt submission and response normalization for Fireworks API calls."""

    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
        default_request_params: Optional[Dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Store model configuration, authentication, and HTTP state."""
        self.api_key = api_key
        self.model_name = model_name
        self.model = f"accounts/fireworks/models/{model_name}"
        self.base_url = base_url or self.BASE_URL
        self.timeout_s = timeout_s
        self.default_request_params = default_request_params or {
            "temperature": 0.6,
            "top_p": 1.0,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def complete_prompt(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        token_budget: int,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a single prompt completion and return the raw response."""
        if token_budget <= 0:
            raise ValueError("token_budget must be a positive integer.")

        message_payload = list(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(token_budget),
            "messages": message_payload,
            **self.default_request_params,
        }

        logger.debug(f"Posting prompt to {self.model} with {int(token_budget)} token budget.")

        response = self.session.post(
            self.base_url,
            headers=self._headers(),
            json=payload,
            timeout=timeout_s or self.timeout_s,
        )
        response.raise_for_status()
        return self.normalize_response(response.json())

    def normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the fields we care about from the raw Fireworks payload."""
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")

        if content is None:
            return

        usage = response.get("usage") or {}
        return {
            "baseline_completion": content,
            "baseline_prompt_tokens": usage.get("prompt_tokens"),
            "baseline_completion_tokens": usage.get("completion_tokens"),
            "baseline_total_tokens": usage.get("total_tokens"),
        }
