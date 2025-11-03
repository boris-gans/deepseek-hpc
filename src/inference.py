"""CLI entry point for DeepSeek inference with local debug mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - helpful runtime message
    torch = None

from . import utils


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek distributed inference entry point."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="File containing newline separated prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store JSONL completions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts per inference step.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-v3.1",
        help="Model identifier used for remote execution.",
    )
    parser.add_argument(
        "--local_debug",
        action="store_true",
        help="Run a CPU-only mock using torch.no_grad() and skip DeepSpeed/NCCL.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def load_prompts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def chunked(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_debug_model():
    if torch is None:
        raise ImportError(
            "PyTorch is required for --local_debug mode. Install torch before running."
        )

    class SimpleDebugModel(torch.nn.Module):  # type: ignore[misc]
        """Tiny module that maps ASCII codes to a scalar score."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(self.proj.weight, 1.0 / 255.0)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # (batch, seq)
            # Compute mean ASCII value per prompt as a toy stand-in for logits.
            return inputs.mean(dim=1, keepdim=True)

    return SimpleDebugModel()


def run_local_debug(
    prompts: Sequence[str],
    *,
    batch_size: int,
    tracker: utils.MetricsTracker,
) -> List[dict]:
    model = _build_debug_model()
    model.eval()

    outputs: List[dict] = []
    start_wall = utils.monotonic_time_s()

    for batch in chunked(prompts, batch_size):
        with utils.timed_phase(tracker, "batch_latency_s"):
            ascii_tensors = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor([ord(ch) for ch in text], dtype=torch.float32)
                    for text in batch
                ],
                batch_first=True,
            )
            with torch.no_grad():
                scores = model(ascii_tensors).squeeze(-1)

        tracker.increment("prompts_processed", len(batch))
        for prompt, score in zip(batch, scores.tolist()):
            outputs.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "debug_completion": f"[local-debug] score={score:.3f}",
                }
            )

    total_wall = utils.monotonic_time_s() - start_wall
    outputs.append(
        {
            "metrics": {
                "total_time_s": total_wall,
                "throughput_qps": tracker.throughput(
                    "prompts_processed", max(total_wall, 1e-6)
                ),
                **tracker.summary(),
            }
        }
    )
    return outputs


def run_distributed_stub(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "Distributed DeepSpeed inference path not yet implemented. "
        "Integrate cluster initialization, tensor parallel groups, and NCCL setup here."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logger = utils.setup_logging(level=args.log_level.upper())
    tracker = utils.MetricsTracker()

    logger.info("Loading prompts from %s", args.input)
    prompts = load_prompts(args.input)
    if not prompts:
        logger.warning("No prompts found in %s. Exiting early.", args.input)
        return

    logger.info(
        "Starting inference (%s mode) with batch_size=%d",
        "local-debug" if args.local_debug else "cluster",
        args.batch_size,
    )

    if args.local_debug:
        try:
            results = run_local_debug(
                prompts,
                batch_size=args.batch_size,
                tracker=tracker,
            )
        except ImportError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    else:
        run_distributed_stub(args)
        return

    if args.output:
        logger.info("Writing outputs to %s", args.output)
        with args.output.open("w") as handle:
            for record in results:
                handle.write(json.dumps(record) + "\n")
    else:
        for record in results:
            logger.info("Result: %s", json.dumps(record))


if __name__ == "__main__":  # pragma: no cover
    main()
