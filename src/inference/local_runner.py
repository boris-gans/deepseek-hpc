"""Local inference runner that populates the prompt DataFrame with baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from .openai_client import OpenAICompletionClient


class LocalInferenceRunner:
    """Iterates over prompts, queries OpenAI, and records completions locally."""

    def __init__(self, client: OpenAICompletionClient, dataframe: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        """Capture the OpenAI client and the DataFrame that will be decorated."""
        self.client = client
        self.dataframe = dataframe

    def run(self) -> None:
        """Execute completions for every prompt row and store metadata."""
        raise NotImplementedError("Local inference execution is not implemented yet.")

    def persist(self, path: Path) -> None:
        """Save the augmented DataFrame to disk for downstream components."""
        raise NotImplementedError("Local inference persistence is not implemented yet.")

    def attach_metrics(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return the DataFrame with any additional latency columns added."""
        raise NotImplementedError("Local inference metrics attachment is not implemented yet.")
