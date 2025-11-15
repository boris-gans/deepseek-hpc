"""Conversion utilities for building the canonical prompt DataFrame."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .prompts import PromptRecord


logger = logging.getLogger(__name__)


class PromptDataFrameBuilder:
    """Creates the shared DataFrame used across local and cluster runs."""

    def __init__(self, prompts: Sequence[PromptRecord] | None) -> None:
        """Capture the prompt records that will populate the table."""
        self._prompts = prompts
        self._df: pd.DataFrame | None = None

    def build(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Materialize an in-memory pandas DataFrame with the canonical schema."""
        if self._prompts is None:
            logger.error("Cannot build dataframe if no prompts have been loaded.")
            raise RuntimeError("Cannot build dataframe if no prompts have been loaded.")
        
        logger.info("Building DataFrame with %d prompts", len(self._prompts))

        rows = [
            {
                "prompt_id": p.prompt_id,
                "variant": p.variant,
                "prompt": p.prompt,
                "token_budget": p.token_budget,
                "source_file": p.source_file,
            }
            for p in self._prompts
        ]

        df = pd.DataFrame(rows)

        # Init metric columns
        df["accuracy"] = None

        self._df = df
        return df

    def persist(self, path: Path) -> None:
        """Write the DataFrame to disk for reuse by other components."""
        if self._df is None:
            raise RuntimeError("DataFrame has not been built yet. Call build() first.")

        path.parent.mkdir(parents=True, exist_ok=True)

        self._df.to_parquet(path, index=False)
        logger.info("Saved prompt DataFrame to disk at %s", path)

    def load_df_from_parquet(self, path: Path):
        if path is None:
            logger.warning("No file path provided to load DataFrame from. Falling back to building DataFrame.")
            return
        
        df = pd.read_parquet(path)
        if not df.empty and len(df) == 40:
            return df
        
        logger.warning("Loaded DataFrame is empty or does not have enough prompts. Falling back to building DataFrame.")
        return
