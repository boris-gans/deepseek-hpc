"""Top-level orchestration pipeline that sequences the parallelism strategies."""

from __future__ import annotations

from typing import Iterable, Sequence

from ..parallelism.strategy import ParallelismStrategy


class DistributedExperimentPipeline:
    """Runs each parallelism strategy and coordinates Slurm submissions."""

    def __init__(self, strategies: Sequence[ParallelismStrategy]) -> None:
        """Capture the ordered list of strategies to execute."""
        self.strategies = list(strategies)

    def run(self) -> None:
        """Execute all strategies sequentially and block until completion."""
        raise NotImplementedError("Pipeline execution is not implemented yet.")

    def run_strategy(self, strategy: ParallelismStrategy) -> None:
        """Execute a single strategy, including job submission and result ingestion."""
        raise NotImplementedError("Single-strategy execution is not implemented yet.")

    def collect_results(self) -> Iterable[dict]:
        """Collect result artifacts emitted by the executed strategies."""
        raise NotImplementedError("Result collection is not implemented yet.")
