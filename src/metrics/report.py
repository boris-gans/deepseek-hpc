"""Reporting helpers that render correctness and performance summaries."""

from __future__ import annotations

from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


class RunSummary:
    """Formats evaluation results into human-friendly artifacts."""

    def __init__(self, dataframe: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        """Capture the DataFrame containing correctness and performance metrics."""
        self.dataframe = dataframe

    def to_markdown(self) -> str:
        """Render the summary into a Markdown table for documentation."""
        raise NotImplementedError("Markdown reporting is not implemented yet.")

    def to_csv(self, path: Path) -> None:
        """Write the summary to disk as a CSV file."""
        raise NotImplementedError("CSV reporting is not implemented yet.")

    def build_plots(self, output_dir: Path) -> None:
        """Generate plots (e.g., correctness vs. latency) for quick inspection."""
        raise NotImplementedError("Plot generation is not implemented yet.")
