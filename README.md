# Distributed Inference of DeepSeek v3.1 on Multi-GPU Clusters

## Overview
Large language models such as DeepSeek v3.1 are too large to fit on a single GPU, so even inference requires multi-GPU, multi-node execution.  
This project implements and benchmarks a distributed inference pipeline using **PyTorch** with **DeepSpeed Inference** (which uses **NCCL** for GPU communication).  
CPU nodes are optionally used for orchestration and preprocessing, allowing us to explore hybrid CPU–GPU performance.

## Objectives
- Run DeepSeek v3.1 inference on 2–3 GPU nodes under **Slurm**.
- Measure "correctness" of the model through comparing a set of 20 prompts with a baseline  
- Measure scaling (throughput / latency / efficiency) from 1 → N nodes.
- Profile compute vs. communication time using **Nsight Systems**, **perf**, and **sacct**.  
- Package everything in an **Apptainer** container for full reproducibility.  
- Produce a short paper and EuroHPC proposal describing results and scaling limits.

## Tech Stack
- **PyTorch** — model runtime  
- **DeepSpeed Inference** — tensor + pipeline parallelism  
- **NCCL** — GPU–GPU communication backend  
- **Slurm** — cluster scheduling  
- **Apptainer** — containerization  
- **Nsight / perf / sacct** — profiling and performance analysis

## Quick Start
```bash
git clone https://github.com/<your-org>/deepseek-hpc.git
cd deepseek-hpc/env
apptainer build deepseek.sif project.def
sbatch slurm/submit_inference.sbatch

# Local debug workflow (CPU only)
python -m src.inference --input data/sample_inputs.txt --local_debug --output results/local_debug.jsonl

# Orchestrate local workers
python -m src.orchestrator --input data/sample_inputs.txt --local_debug --num_workers 2 --dispatch_size 4
```

## Development Skeleton
- `src/inference.py` exposes the CLI entry point. Pass `--local_debug` to bypass DeepSpeed/NCCL setup and run a lightweight PyTorch-only flow that verifies I/O, batching, and logging.
- `src/orchestrator.py` simulates distributed request handling with threads so you can refine orchestration logic before running under Slurm.
- `src/utils.py` provides shared logging and performance tracking helpers (throughput, latency, CPU timings).
- `results/` is where we store the system performance, feeding directly into the "performance and scaling analysis" section of the paper. It answers speed, effecienty and scalability
- `eval/`is where we perform model quality evaluation. It answers model correctness, and if precision or sharding affect model quality
