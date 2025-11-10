"""Conversion utilities for building the canonical prompt DataFrame."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .prompts import PromptRecord



class PromptDataFrameBuilder:
    """Creates the shared DataFrame used across local and cluster runs."""

    def __init__(self, prompts: Sequence[PromptRecord]) -> None:
        """Capture the prompt records that will populate the table."""
        self._prompts = prompts

    def build(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Materialize an in-memory pandas DataFrame with the canonical schema."""
        raise NotImplementedError("Prompt DataFrame construction is not implemented yet.")

    def persist(self, path: Path) -> None:
        """Write the DataFrame to disk for reuse by other components."""
        raise NotImplementedError("Prompt DataFrame persistence is not implemented yet.")
