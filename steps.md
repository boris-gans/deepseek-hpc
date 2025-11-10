# Distributed Inference Build Plan

This document captures the class-based layout and execution flow for the revised pipeline described in `README.md`. It assumes a local orchestration node plus a Slurm-managed GPU cluster.

## 1. Prompt Assets (local CPU)
- **`data/prompts_2k.txt` + `data/prompts_4k.txt`**: Two curated text files, 20 prompts each, targeting ~2k and ~4k token budgets.
- **`src/data/prompts.py`**
  - `PromptRecord`: dataclass with `prompt_id`, `variant` (`2k`/`4k`), `prompt`, `token_budget`.
  - `PromptFileSet`: validates both files exist, enforces 20 prompts per file, attaches IDs (shared across variants).
  - `PromptRepository`: exposes `load_all()` → `list[PromptRecord]`, caches metadata for downstream consumers.

## 2. Prompt Table & Local Baseline
- **`src/data/table.py`**
  - `PromptDataFrameBuilder`: converts `PromptRecord` objects into a pandas `DataFrame`.
  - Columns: `prompt_id`, `variant`, `prompt`, `token_budget`, `source_file`, `local_completion`, `local_latency_ms`, `status`.
- **`src/inference/openai_client.py`**
  - `OpenAICompletionClient`: wraps the OpenAI Python SDK to call the remote Llama 3.3 70B endpoint (hosted outside the cluster).
  - Handles retries, rate limiting, and response normalization.
- **`src/inference/local_runner.py`**
  - `LocalInferenceRunner`: iterates over the DataFrame, calls `OpenAICompletionClient`, and writes `local_completion` + timing back into the DataFrame.
  - Persists the table to `results/baseline_prompts.parquet` for reuse by distributed runs.

## 3. Distributed Experiment Orchestration (Slurm)
- **`src/parallelism/strategy.py`**
  - `ParallelismStrategy` (abstract): defines `name`, `slurm_constraints`, `deepspeed_args`, `postprocess(df)` and `expected_jobs`.
  - Concrete subclasses align with the four README techniques:
    1. `TensorParallelAcrossNodes` (TP-only across nodes).
    2. `PipelineParallelAcrossNodes` (PP-only across nodes).
    3. `TensorParallelSingleNode` (TP-only single node).
    4. `HybridTensorPipeline` (TP intra-node + PP inter-node).
- **`src/slurm/job_factory.py`**
  - `SlurmConfig`: dataclass describing nodes, GPUs, walltime, env, container, output paths.
  - `SlurmJobFactory`: builds configs per strategy run (40 prompts = 20×2 variants). Injects prompt shard paths and output locations.
- **`src/slurm/job_manager.py`**
  - `SlurmJobManager`: submits jobs, monitors status via `sacct`/`squeue`, and triggers callback once logs land locally.
  - Responsible for “dynamic config”: after each job finishes, it asks the relevant strategy for the next config (allowing parameter sweeps or retries).
- **`src/orchestration/pipeline.py`**
  - `DistributedExperimentPipeline`: runs strategies sequentially, ensures each processes both prompt variants, and hands results to the `ResultCollector`.

## 4. Result Collection & Metrics (local CPU)
- **`src/results/collector.py`**
  - `ResultCollector`: loads JSON/Parquet artifacts produced on the cluster, normalizes schema, and merges them back into the master DataFrame under columns `strategy`, `rank`, `gpu_id`, `cluster_completion`, `cluster_latency_ms`, `trace_path`.
- **`src/metrics/correctness.py`**
  - `CorrectnessEvaluator`: after all four strategies complete, computes your “correctness” metric per prompt/strategy pair (uses the baseline completions stored earlier).
- **`src/metrics/report.py`**
  - `RunSummary`: emits CSV/Markdown summaries plus plots (latency vs. correctness, throughput vs. token length).

## 5. Execution Steps
1. **Prepare prompts locally** using `PromptFileSet`. Persist canonical IDs to keep rows aligned between 2k and 4k variants.
2. **Build the prompt DataFrame** (`PromptDataFrameBuilder`). Store to disk before inference so downstream scripts can reload the same ordering.
3. **Run baseline OpenAI inference** with `LocalInferenceRunner`; update `local_completion` + metrics columns.
4. **Launch distributed strategies sequentially** via `DistributedExperimentPipeline`:
   - For each strategy, instantiate `ParallelismStrategy` subclass.
   - `SlurmJobFactory` generates the job script/config dynamically (include specific TP/PP/DP settings described in README).
   - `SlurmJobManager` submits, monitors, and triggers data ingestion when outputs arrive.
5. **Ingest job outputs** with `ResultCollector`, append to the shared DataFrame, and checkpoint to `results/run_<timestamp>.parquet`.
6. **Compute correctness metrics** (outside the cluster) and export a summary (CSV/Markdown) comparing all four strategies.