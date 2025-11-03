"""Local orchestration for distributed inference workflows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from . import utils
from .inference import chunked, load_prompts, run_distributed_stub, run_local_debug


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate DeepSeek inference across workers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prompts file consumed by the orchestrator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file aggregating all responses.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of concurrent workers to simulate.",
    )
    parser.add_argument(
        "--dispatch_size",
        type=int,
        default=8,
        help="Number of prompts dispatched per worker submission.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size passed through to inference.py.",
    )
    parser.add_argument(
        "--local_debug",
        action="store_true",
        help="Route requests to inference.py --local_debug mode.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def _split_for_workers(prompts: Sequence[str], dispatch_size: int) -> Iterable[List[str]]:
    for batch in chunked(prompts, dispatch_size):
        yield batch


def _run_worker(
    prompts: Sequence[str],
    *,
    batch_size: int,
) -> Tuple[List[dict], dict]:
    tracker = utils.MetricsTracker()
    results = run_local_debug(prompts, batch_size=batch_size, tracker=tracker)

    completions = [record for record in results if "metrics" not in record]
    aggregated = next(
        (record["metrics"] for record in results if "metrics" in record), {}
    )
    return completions, aggregated


def orchestrate_local_debug(
    prompts: Sequence[str],
    *,
    num_workers: int,
    dispatch_size: int,
    batch_size: int,
    logger,
) -> List[dict]:
    completions: List[dict] = []
    worker_metrics: List[dict] = []
    start = utils.monotonic_time_s()

    logger.info(
        "Dispatching %d prompts to %d workers (dispatch_size=%d, batch_size=%d)",
        len(prompts),
        num_workers,
        dispatch_size,
        batch_size,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _run_worker,
                batch,
                batch_size=batch_size,
            ): batch
            for batch in _split_for_workers(prompts, dispatch_size)
        }

        for future in as_completed(futures):
            batch = futures[future]
            try:
                worker_completions, metrics = future.result()
            except Exception as exc:  # pragma: no cover - surface to user
                if isinstance(exc, ImportError):
                    raise
                logger.exception(
                    "Worker failed while processing batch of size %d: %s",
                    len(batch),
                    exc,
                )
                continue

            logger.debug(
                "Worker completed %d prompts with throughput %.2fqps",
                len(worker_completions),
                metrics.get("throughput_qps", 0.0),
            )
            completions.extend(worker_completions)
            worker_metrics.append(metrics)

    total_elapsed = max(utils.monotonic_time_s() - start, 1e-6)
    aggregate = {
        "total_elapsed_s": total_elapsed,
        "num_prompts": len(completions),
        "avg_worker_throughput_qps": (
            sum(m.get("throughput_qps", 0.0) for m in worker_metrics)
            / max(len(worker_metrics), 1)
        ),
        "avg_batch_latency_s": (
            sum(
                m.get("latency", {}).get("batch_latency_s", 0.0)
                for m in worker_metrics
            )
            / max(len(worker_metrics), 1)
        ),
        "overall_throughput_qps": len(completions) / total_elapsed,
    }
    completions.append({"metrics": aggregate})
    return completions


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logger = utils.setup_logging(level=args.log_level.upper(), name="orchestrator")

    prompts = load_prompts(args.input)
    if not prompts:
        logger.warning("No prompts found in %s. Exiting early.", args.input)
        return

    if args.local_debug:
        try:
            records = orchestrate_local_debug(
                prompts,
                num_workers=args.num_workers,
                dispatch_size=args.dispatch_size,
                batch_size=args.batch_size,
                logger=logger,
            )
        except ImportError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    else:
        logger.info("Cluster mode requested. Handing off to distributed stub.")
        run_distributed_stub(args)
        return

    if args.output:
        logger.info("Writing aggregated outputs to %s", args.output)
        with args.output.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    else:
        for record in records:
            logger.info("Result: %s", json.dumps(record))


if __name__ == "__main__":  # pragma: no cover
    main()
