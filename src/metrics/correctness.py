"""Correctness metric scaffolding for comparing strategy outputs."""

from __future__ import annotations

from typing import Dict, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


class CorrectnessEvaluator:
    """Computes custom correctness metrics per prompt and strategy."""

    def __init__(self, baseline_column: str = "local_completion") -> None:
        """Store the DataFrame column that represents the baseline reference."""
        self.baseline_column = baseline_column

    def evaluate(self, dataframe: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return a DataFrame augmented with correctness scores."""
        raise NotImplementedError("Correctness evaluation is not implemented yet.")

    def score_prompt(self, baseline: str, candidate: str) -> float:
        """Score a single prompt's candidate response."""
        raise NotImplementedError("Prompt scoring is not implemented yet.")

    def summarize(self, dataframe: "pd.DataFrame") -> Dict[str, float]:  # type: ignore[name-defined]
        """Produce aggregate correctness metrics for reporting."""
        raise NotImplementedError("Correctness summarization is not implemented yet.")
