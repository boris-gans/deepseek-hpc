"""Utilities for logging and lightweight performance tracking."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import DefaultDict, Dict, Iterator, List, Optional


def setup_logging(
    *,
    name: str = "deepseek",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure a root logger shared by CLI entry points."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)7s | %(threadName)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured (level=%s, file=%s)", level, log_file)
    return logger


@dataclass
class MetricsTracker:
    """Tracks latency and throughput metrics for inference requests."""

    _latencies: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _counters: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_latency(self, name: str, duration_s: float) -> None:
        with self._lock:
            self._latencies[name].append(duration_s)

    def increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._counters[name] += count

    def throughput(self, counter_name: str, total_time_s: float) -> float:
        value = self._counters[counter_name]
        if total_time_s <= 0:
            return 0.0
        return value / total_time_s

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return percentile-friendly aggregates consumers can log."""
        with self._lock:
            return {
                "latency": {
                    name: mean(samples) if samples else 0.0
                    for name, samples in self._latencies.items()
                },
                "counters": dict(self._counters),
            }


@contextmanager
def timed_phase(tracker: MetricsTracker, name: str) -> Iterator[None]:
    """Context manager that records phase duration in seconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        tracker.record_latency(name, duration)


def monotonic_time_s() -> float:
    """Expose monotonic time helper so callers can compute elapsed durations."""
    return time.perf_counter()
