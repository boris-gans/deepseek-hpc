"""Prompt ingestion primitives for the distributed inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


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
        """Ensure both prompt files exist and contain the 20 prompts each."""

        logger.info(
            "Validating prompt files (2k=%s, 4k=%s)", self.path_2k, self.path_4k
        )

        for label, p in [("2k prompt", self.path_2k), ("4k prompt", self.path_4k)]:
            if not p.exists():
                logger.error("Path %s does not exist", p)
                raise FileNotFoundError(f"[PromptFileSet] {label} file does not exist: {p}")

            if not p.is_file():
                logger.error("Path %s is not a file.", p)
                raise ValueError(f"[PromptFileSet] {label} path is not a file: {p}")

            with p.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) != 20:
                logger.error("%s must contain exactly 20 lines, but has %d: %s", label, len(lines), p)
                raise ValueError(
                    f"[PromptFileSet] {label} must contain exactly 20 lines, but has {len(lines)}: {p}"
                )
            
            logger.debug("[%s] Loaded %d prompts from %s", label, len(lines), p)

    def _load_records_from_file(self, p: Path, variant: str, token_budget: int) -> List[PromptRecord]:
        records: List[PromptRecord] = []

        logger.debug("Reading prompt records for variant %s from %s", variant, p)
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                text = line.rstrip("\n")

                record = PromptRecord(
                    prompt_id=f"{variant}_{idx}",
                    variant=variant,
                    prompt=text,
                    token_budget=token_budget,
                    source_file=p,
                )
                records.append(record)

        logger.debug(
            "Loaded %d %s prompt records from %s", len(records), variant, p
        )
        return records

    def iter_records(self) -> Dict[str, List[PromptRecord]]:
        """Yield `PromptRecord` objects for both prompt variants."""
        records_2k = self._load_records_from_file(self.path_2k, variant="2k", token_budget=2000)
        records_4k = self._load_records_from_file(self.path_4k, variant="4k", token_budget=4000)

        return {
            "2k": records_2k, 
            "4k": records_4k,
        }


class PromptRepository:
    """Loads prompt records and caches them for downstream consumers."""

    def __init__(self, file_set: PromptFileSet) -> None:
        """Initialize the repository with a validated file set."""
        self._file_set = file_set

        self.records_2k: List[PromptRecord] = []
        self.records_4k: List[PromptRecord] = []

        logger.debug(
            "PromptRepository initialized with files: 2k=%s, 4k=%s",
            file_set.path_2k,
            file_set.path_4k,
        )


    def load_all(self) -> List[PromptRecord]:
        """Load every prompt record into memory."""
        logger.info("Loading prompt records from file set.")

        self._file_set.validate()

        records_by_variant = self._file_set.iter_records()
        self.records_2k = records_by_variant["2k"]
        self.records_4k = records_by_variant["4k"]

        logger.info(
            "Cached %d 2k prompts and %d 4k prompts",
            len(self.records_2k),
            len(self.records_4k),
        )

        return self.records_2k + self.records_4k


    def get_by_id(self, prompt_id: str) -> Sequence[PromptRecord]:
        """Fetch all variants associated with a specific prompt identifier."""
        raise NotImplementedError("Prompt lookup is not implemented yet.")
