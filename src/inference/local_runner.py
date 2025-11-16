"""Local inference runner that populates the prompt DataFrame with baselines."""

from __future__ import annotations

import logging
import requests
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .fireworksAI_client import FireworksAICompletionClient


logger = logging.getLogger(__name__)


class LocalInferenceRunner:
    """Iterates over prompts, queries OpenAI, and records completions locally."""

    BASELINE_COLUMNS = [
        "baseline_completion",
        "baseline_prompt_tokens",
        "baseline_completion_tokens",
        "baseline_total_tokens",
    ]

    def __init__(self, client: FireworksAICompletionClient, dataframe: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """Capture the OpenAI client and the DataFrame that will be decorated."""
        self.client = client
        self.df = dataframe
        self.df_acc = pd.DataFrame | None

    def run(self, limit: Optional[int] = None, variant: Optional[str] = None, system_prompt: Optional[str] = None) -> pd.DataFrame:
        """Execute completions for every prompt row and store metadata."""

        required_columns = {"prompt", "token_budget", "variant"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {', '.join(sorted(missing))}"
            )
        
        working_df = self.df.copy(deep=True)
        for column in self.BASELINE_COLUMNS:
            if column not in working_df.columns:
                working_df[column] = None

        target_df = working_df
        if variant is not None:
            # allowed = set(variants)
            target_df = working_df[working_df["variant"] == variant]

        if limit is not None:
            if limit <= 0:
                limit = None
                logger.warning("Invalid limit specified, ignoring parameter.")
            else:
                target_df = target_df.head(limit)


        logger.info(
            "Submitting %d prompts to Fireworks (variants=%s, limit=%s)",
            len(target_df),
            variant if variant else "all",
            limit if limit is not None else "none",
        )

        for index, row in target_df.iterrows():
            prompt = row["prompt"]
            token_budget = int(row["token_budget"])

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                logger.debug("Sending request to Fireworks for prompt_id=%s", row.get("prompt_id"))
                response = self.client.complete_prompt(
                    messages=messages,
                    token_budget=token_budget
                )
            except requests.RequestException as exc:
                logger.exception("Fireworks request failed for prompt_id=%s", row.get("prompt_id"))
                raise

            # Check if normalization failed
            if response is None:
                logger.warning("Unable to normalize response data for prompt_id=%s, skipping...", row.get("prompt_id"))
                continue

            # Add results to df
            for column, value in response.items():
                working_df.at[index, column] = value

        self.df_acc = working_df
        return working_df

    def persist(self, path: Path) -> None:
        """Save the augmented DataFrame to disk for downstream components."""
        if self.df_acc is None:
            raise RuntimeError("DataFrame has not been built yet. Call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.df_acc.to_parquet(path, index=False)
        logger.info("Saved DataFrame with baseline responses to disk at %s", path)

    def attach_metrics(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return the DataFrame with any additional latency columns added."""
        raise NotImplementedError("Local inference metrics attachment is not implemented yet.")
