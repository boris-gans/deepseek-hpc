"""Client wrapper around the OpenAI SDK for local baseline inference."""

from __future__ import annotations

from typing import Any, Dict, List


class OpenAICompletionClient:
    """Handles prompt submission and response normalization for OpenAI calls."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Store model configuration and authentication material."""
        self.model_name = model_name
        self.api_key = api_key

    def complete_prompt(self, prompt: str) -> Dict[str, Any]:
        """Execute a single prompt completion and return the raw response."""
        raise NotImplementedError("OpenAI completion is not implemented yet.")

    def complete_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Submit a batch of prompts and collect their responses."""
        raise NotImplementedError("OpenAI batch completion is not implemented yet.")

    def normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the fields we care about from the raw OpenAI payload."""
        raise NotImplementedError("OpenAI response normalization is not implemented yet.")
