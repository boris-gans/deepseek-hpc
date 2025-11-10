"""Prompt ingestion primitives for the distributed inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class PromptRecord:
    """Container for a single prompt variant and associated metadata."""

    prompt_id: str
    variant: str
    prompt: str
    token_budget: int
    source_file: Path | None = None


class PromptFileSet:
    """Validates and exposes the canonical 2k and 4k prompt files."""

    def __init__(self, path_2k: Path, path_4k: Path) -> None:
        """Store references to the prompt files chosen for this run."""
        self.path_2k = path_2k
        self.path_4k = path_4k

    def validate(self) -> None:
        """Ensure both prompt files exist and contain the expected counts."""
        raise NotImplementedError("Prompt file validation is not implemented yet.")

    def iter_records(self) -> Iterable[PromptRecord]:
        """Yield `PromptRecord` objects for both prompt variants."""
        raise NotImplementedError("Prompt record iteration is not implemented yet.")


class PromptRepository:
    """Loads prompt records and caches them for downstream consumers."""

    def __init__(self, file_set: PromptFileSet) -> None:
        """Initialize the repository with a validated file set."""
        self._file_set = file_set

    def load_all(self) -> List[PromptRecord]:
        """Load every prompt record into memory."""
        raise NotImplementedError("Prompt loading is not implemented yet.")

    def get_by_id(self, prompt_id: str) -> Sequence[PromptRecord]:
        """Fetch all variants associated with a specific prompt identifier."""
        raise NotImplementedError("Prompt lookup is not implemented yet.")
