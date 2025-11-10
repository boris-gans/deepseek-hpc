# Distributed Inference of Llama 3.3 70B on Multi-GPU Clusters

## Overview
Large language models such as Llama 3.3 70B are too large to fit on a single GPU, so even inference requires multi-GPU, multi-node execution.  
This project implements and benchmarks a distributed inference pipeline using **PyTorch** with **DeepSpeed Inference** (which uses **NCCL** for GPU communication).  
CPU nodes are optionally used for orchestration and preprocessing, allowing us to explore hybrid CPU–GPU performance.

## Objectives
- Run Llama 3.3 70B inference on 2–3 GPU nodes under **Slurm**.
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

## Model Hyperparameters

* **L, H, D**: `L=80`, `H=64`, `D=128` (hidden size `8192`).
* **KV per token (BF16/FP16)**: `~2.5 MiB`.
  * KV@2k: **5.0 GiB** total; KV@4k: **10.0 GiB** total (batch=1).
  * **Per-GPU KV** = total_KV / (**TP×PP**).
* **Weights** (BF16): 70e9 x 2B = **~130.4 GiB** total = **~140 GB** decimal.
  * **Per-GPU weights** = total_weights / (**TP×PP**).
* **Other activations** (non-KV) budget: **~2 GiB/GPU** (tune with measurements).
* **Overhead** budget: **~6 GiB/GPU** (CUDA/NCCL/allocator/workspaces).
  * **Batch-size > 1** Activation memory (including KV) scales lineary with batch-size


This table shows the **per GPU memory**, assuming 8 GPUs (TPxPP always equals 8, regardless of cluster config).


| Component         | Formula                     | Value (GiB) |
| ----------------- | --------------------------- | ----------- |
| Weights           | 130.4 / (TP×PP) = 130.4 / 8 | **16.30**   |
| KV @ **2k**       | 5.0 / 8                     | **0.625**   |
| KV @ **4k**       | 10.0 / 8                    | **1.25**    |
| Other activations | fixed                       | **2.0**     |
| Overhead          | fixed                       | **6.0**     |
| **Total @ 2k**    | sum                         | **24.93**   |
| **Total @ 4k**    | sum                         | **25.55**   |

<hr>

## Cluster Configurations
We'll explore multiple cluster configurations and parallelism techniques to identify bottlenecks and improve performance. We'll deploy a cluster on [PrimeIntellect](https://docs.primeintellect.ai/introduction), with 8 H100 Nvidia GPUs (80GB each). 

<hr>

### TP-only across nodes:

    a. Nodes/GPUs: 8 nodes x 1 GPU each

    b. Parallelism: TP = 8, PP = 1, DP = 1

    c. Hypothesis of issues: Low GPU utilization, NCCL time dominates

TP requires sync at every layer, meaning lots of network traffic. If this is done across nodes it will create huge latency/idle bubbles.

**Cluster Topology:**

                         ┌─────────────────────────────────────────────────────────────┐
                         │            DeepSeek-70B (Cross-node TP only)               │
                         └─────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────┐
                           │        Input Prompt Batch        │
                           └──────────────────────────────────┘
                                              │
                                              ▼
         ┌──────────────────────────────────────────────────────────────────────────────┐
         │               Tensor Parallel Group (size = 8, across 8 nodes)               │
         │------------------------------------------------------------------------------│
         │ Node0 (GPU0) ─┐                                                             │
         │ Node1 (GPU1) ─┼───╮                                                         │
         │ Node2 (GPU2) ─┼───┼─── NCCL all-reduce / all-gather (per layer) ────────────┤
         │ Node3 (GPU3) ─┼───┤                                                         │
         │ Node4 (GPU4) ─┼───┤   <─── High-latency InfiniBand / Ethernet links ───>    │
         │ Node5 (GPU5) ─┼───┤                                                         │
         │ Node6 (GPU6) ─┼───┤                                                         │
         │ Node7 (GPU7) ─┘                                                             │
         └──────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Final Model Output    │
                                  └────────────────────────┘


<hr>

### PP-only across nodes:

    a. Nodes/GPUs: 8 nodes x 1 GPU each

    b. Parallelism: TP = 1, PP = 8, DP = 1

    c. Expected: good stability, the overall latency will be shaped by pipeline bubbles

Only pipeline traffic across nodes, meaning its done once per micro-batch.

**Cluster Topology:**

                            ┌─────────────────────────────────────────────────────────────┐
                            │             DeepSeek-70B (PP=8, TP=1) Cluster               │
                            └─────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                            ┌──────────────────────────────────┐
                            │        Input Prompt Batch        │
                            └──────────────────────────────────┘
                                                │
                                                ▼
    ┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
    │         Node 0 (GPU0)       │         Node 1 (GPU1)       │         Node 2 (GPU2)       │         Node 3 (GPU3)       │
    │ Layers 0–N/8                │ Layers N/8–2N/8             │ Layers 2N/8–3N/8            │ Layers 3N/8–4N/8            │
    │ Pipeline Stage 0            │ Pipeline Stage 1            │ Pipeline Stage 2            │ Pipeline Stage 3            │
    └──────────────┬──────────────┴──────────────┬──────────────┴──────────────┬──────────────┴──────────────┬──────────────┘
                │                             │                             │                             │
                ▼                             ▼                             ▼                             ▼
    ┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
    │         Node 4 (GPU4)       │         Node 5 (GPU5)       │         Node 6 (GPU6)       │         Node 7 (GPU7)       │
    │ Layers 4N/8–5N/8            │ Layers 5N/8–6N/8            │ Layers 6N/8–7N/8            │ Layers 7N/8–N               │
    │ Pipeline Stage 4            │ Pipeline Stage 5            │ Pipeline Stage 6            │ Pipeline Stage 7            │
    └──────────────┬──────────────┴──────────────┬──────────────┴──────────────┬──────────────┴──────────────┬──────────────┘
                │                             │                             │                             │
                │─────── Activations / KV-cache (Inter-node transfers) ─────│
                                                ▼
                                    ┌────────────────────────┐
                                    │  Final Model Output    │
                                    └────────────────────────┘

<hr>

### TP-only single node:

    a. Nodes/GPUs: 1 node x 8 GPUs

    b. Parallelism: TP = 8, PP = 1, DP = 1

    c. Expected: Strong baseline with high utilization

The per-layer sync now occurs on NVLink within a node, ensuring traffic is must faster at each sync (per layer).

**Cluster Topology:**

                         ┌─────────────────────────────────────────────────────────────┐
                         │          DeepSeek-70B (Single-node TP=8) Cluster            │
                         └─────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────┐
                           │        Input Prompt Batch        │
                           └──────────────────────────────────┘
                                              │
                                              ▼
         ┌──────────────────────────────────────────────────────────────────────────────┐
         │                       Node 0 — NVLink / NVSwitch Fabric                      │
         │------------------------------------------------------------------------------│
         │   GPU0 ─┐                                                                    │
         │   GPU1 ─┼───╮                                                                │
         │   GPU2 ─┼───┼─── NCCL all-reduce / all-gather (per layer, intra-node) ───────┤
         │   GPU3 ─┼───┤                                                                │
         │   GPU4 ─┼───┤                                                                │
         │   GPU5 ─┼───┤                                                                │
         │   GPU6 ─┼───┤                                                                │
         │   GPU7 ─┘   │                                                                │
         │------------------------------------------------------------------------------│
         │ All communication on NVLink/NVSwitch (~900 GB/s)                             │
         └──────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Final Model Output    │
                                  └────────────────────────┘

<hr>

### Hybrid TP-intra + PP-inter:

    a. Nodes/GPUs: 2 nodes x 4 GPUs

    b. Parallelism: TP = 4 (within node), PP = 2 (across nodes), DP = 1

    c. Expected: The best combination of latency and throughput, with a healthy comm/comp ratio

The hybrid setup allows us to make good use of both the NVLink and the large quanitiy of nodes we have available.

**Cluster Topology:**

                         ┌─────────────────────────────────────────────────────────────┐
                         │                DeepSeek-70B Inference Cluster               │
                         └─────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────┐
                           │        Input Prompt Batch        │
                           └──────────────────────────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        │                                           │
                        ▼                                           ▼
             ┌─────────────────────┐                    ┌─────────────────────┐
             │   Node 0 (Stage 0)  │                    │   Node 1 (Stage 1)  │
             │ Layers 0-N/2        │                    │ Layers N/2-N        │
             │ Tensor Parallel = 4 │                    │ Tensor Parallel = 4 │
             └─────────────────────┘                    └─────────────────────┘
                        │                                           │
                        ▼                                           ▼
        ┌──────────────────────────────┐             ┌──────────────────────────────┐
        │   GPU0 ──┐                   │             │   GPU4 ──┐                   │
        │   GPU1 ──┼── NVLink/NCCL ────┼─ TP Group   │   GPU5 ──┼── NVLink/NCCL ────┼─ TP Group
        │   GPU2 ──┼─── intra-node ────┤             │   GPU6 ──┼─── intra-node ────┤
        │   GPU3 ──┘ communication     │             │   GPU7 ──┘ communication     │
        └──────────────────────────────┘             └──────────────────────────────┘
                        │                                           │
                        │<────────────  Pipeline  ────────────────►│
                        │         (activations, KV-cache)          │
                        ▼                                           ▼
                 ┌────────────────┐                         ┌────────────────┐
                 │ Partial Output │                         │ Final Output   │
                 └────────────────┘                         └────────────────┘



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


## Resources
- **Model:** https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- **Hosting:** https://app.primeintellect.ai/dashboard/clusters?gpu_type=H100_80GB&image=ubuntu_22_cuda_12&location=Cheapest&quantity=8

