# Report

## 1. Introduction and Motivation
Large language models frequently exceed the memory capacity of a single GPU, which makes even inference a distributed systems problem. This project is a hands-on exploration of running a medium-size model (OpenLLaMA 7B/3B weight splits) across multiple GPUs and nodes under Slurm, using pipeline parallelism as the primary sharding strategy. Although the target model is modest compared to frontier LLMs, the intent is to develop a reproducible playbook for scaling larger models on constrained academic clusters. The work emphasizes: (a) reliable containerized execution with Apptainer, (b) reproducible orchestration via Slurm, (c) explicit control of device mapping and offload behavior to work within tight GPU memory limits, and (d) metrics collection for later performance analysis.

## 2. System Overview
- **Runtime entrypoint**: `run_distributed_inference.py` orchestrates two pipeline stages (ranks 0 and 1). It partitions transformer layers and loads only the assigned slice onto each rank’s GPU, offloading unused layers to CPU to reduce device memory pressure. Model weights and tokenizer are assumed to be pre-staged on the cluster (no Hugging Face downloads at runtime).
- **Submission stack**: `slurm/submit.sbatch` drives the job submission, loads the Apptainer module, and launches `slurm/run.sh` inside the container on every task using `srun`. The script binds a scratch-backed workspace to `/tmp/workspace` in the container and binds the code repo to `/app`.
- **Container strategy**: We rely on a prebuilt PyTorch Apptainer image (e.g., `/home/user49/scratch/group1/appainter/appainter.sif`). At runtime, `slurm/run.sh` bootstraps Python packages into a shared venv under `/tmp/workspace/.venv`, avoiding torch/CUDA wheels so the container’s native GPU stack is used.
- **Data and prompts**: Prompts are staged under `slurm/prompts.jsonl` and copied into the shared workspace before launch. Inputs are small general-knowledge prompts (~2k tokens) used for correctness and throughput checks.
- **Model storage**: The model is pre-downloaded locally and copied to cluster scratch, then bind-mounted into the container (e.g., `--bind "/home/$USER/scratch/group1/models/openllama-7b:/workspace/model"`). The experiment config references this path so runtime never touches the network.

## 3. Cluster and Filesystem Layout
- **Host paths (before submission)**:
  - Code: `/home/$USER/projects/distributed-inference/`
  - Scratch runtime: `/home/$USER/scratch/group1/pipeline_run/` (bound to `/tmp/workspace`)
  - Model: `/home/$USER/scratch/group1/models/openllama-7b/` (bound into the container, commonly `/workspace/model`)
- **Container view during job**:
  - `/app/`: bind-mounted repo containing `run_distributed_inference.py`, `slurm/run.sh`, configs
  - `/tmp/workspace/`: shared experiment root (configs, outputs, offload dirs, staged prompts)
  - `/workspace/model`: bound model directory containing config, tokenizer, and weight shards
- **Shared outputs**: `/tmp/workspace/outputs/` collects per-rank logs, completion JSONL, and future profiling output.

## 4. Launch Flow and Configuration
### 4.1 Slurm submission (`slurm/submit.sbatch`)
- Loads `apptainer` and `cuda` modules.
- Sets defaults: 2 nodes, 1 task per node, 1 GPU per task, 4 CPUs per task, 30-minute wall-clock, GPU partition.
- Binds project scratch to `/tmp/workspace` and code repo to `/app`.
- Stages configs and prompts from `slurm/` into the workspace:
  - `exp_config.json` → `/tmp/workspace/exp_config.json`
  - `ds_config.json` → `/tmp/workspace/ds_config.json`
  - `prompts.jsonl` → `/tmp/workspace/prompts.jsonl`
- Uses `srun apptainer exec --nv ... bash /app/slurm/run.sh` to start one container per rank.

### 4.2 Runtime launcher (`slurm/run.sh`)
- Sets torch/NCCL env, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, and sane defaults (`OMP_NUM_THREADS=4`, NCCL debug flags).
- Bootstraps Python deps into `/tmp/workspace/.venv` with a lock to avoid races; installs deepspeed/transformers/tokenizers/etc. **without** torch/CUDA wheels, so the container’s CUDA-enabled torch is used.
- Ensures `TRITON_CACHE_DIR` and pip cache live under the shared workspace.
- Runs the base command: `python /app/run_distributed_inference.py --exp_config ... --ds_config ... --output_dir ...` per rank, with per-rank logging to `/tmp/workspace/outputs/rank_<r>.log`.

### 4.3 Experiment config (`slurm/exp_config.json`)
- Example fields:
  - `model.model_path`: `/tmp/workspace/model` (or `/workspace/model` depending on bind)
  - `model.model_name`: descriptive label (not used for downloads)
  - `inputs.prompt_path`: `/tmp/workspace/prompts.jsonl`
  - `outputs.output_dir`: `/tmp/workspace/outputs`
  - Inference knobs: `max_new_tokens`, `temperature`, `top_p`, `batch_size`
  - Parallelism info is primarily driven by DeepSpeed config below.

### 4.4 DeepSpeed config (`slurm/ds_config.json`)
- Specifies `pipeline_parallel_size` and `tensor_parallel_size` (pipeline is actually enforced by world size in `run_distributed_inference.py`, currently a two-stage split).
- Training fields are mostly placeholders; the script uses the `distributed_training` section only to cross-check pipeline size vs world size.

## 5. Model Loading and Partitioning Logic (`run_distributed_inference.py`)
- **Local-only loading**: `model_path` is mandatory in `exp_config`. The script loads tokenizer and model with `local_files_only=True`; no HF network calls are made.
- **Device mapping per rank**: The model has 26 transformer layers (OpenLLaMA 7B/3B variant). With two pipeline stages, rank 0 takes layers `[0,13)` and rank 1 takes `[13,26)`. Device map construction:
  - Layers assigned to the local rank are placed on that rank’s GPU (`torch.device("cuda", LOCAL_RANK)`).
  - Unused layers are assigned to CPU, and state dict offload writes to `/tmp/workspace/outputs/offload_rank_<r>/`.
  - Embeddings live on stage 0; final norm and `lm_head` live on stage 1. (If desired, these mappings can be moved to GPU on stage 0, but current mapping keeps unused components on CPU to save device memory.)
- **Memory-conscious load**: Uses `low_cpu_mem_usage=True` plus offload to minimize RAM pressure during load. Only the kept slice is moved into the active module; the rest remains offloaded/CPU.
- **Execution path**:
  - Rank 0: tokenize prompts, run embedding + first-half layers, send activations to rank 1.
  - Rank 1: receive activations, run second-half layers + norm + head, generate tokens, send next-token IDs back to rank 0.
  - Generation is greedy, micro-batch size 1, BF16 dtype.
- **No downloads**: All paths reference the bound model directory; HF cache/env setup has been removed.

## 6. Data and Prompts
- Prompts are stored in `slurm/prompts.jsonl` (staged into `/tmp/workspace/prompts.jsonl`). They are short general-knowledge prompts (~20 total) around 2k tokens each. The script expects JSONL with a `prompt` field per line.
- For weak/strong scaling experiments, the prompt set can be subsetted or duplicated; batch size defaults to 1 to simplify pipeline flow.

## 7. Hardware and Container Notes
- **GPU binding**: Jobs use `apptainer exec --nv` to expose host NVIDIA drivers. Additional driver library binding was added in submission scripts to ensure `libnvidia-ml.so` is visible; `LD_LIBRARY_PATH` is set accordingly when needed.
- **GPU topology**: Target nodes expose NVIDIA T4 (example `nvidia-smi` shows driver 550.144.06, CUDA 12.4). Each job allocates one GPU per node; pipeline size is tied to the number of tasks.
- **Storage**: Runtime data and caches live under `/home/$USER/scratch/group1/pipeline_run/` to stay off NFS home. Offload directories for unused layers and state dict sharding are placed under `outputs/offload_rank_*`.
- **Networking**: Master address/port is computed in `submit.sbatch` (`MASTER_ADDR` via `scontrol` hostname, `MASTER_PORT=29500` default). Distributed init uses `env://` with backend `nccl` when GPUs are visible, otherwise `gloo`.

## 8. Experimental Plan (Design)
- **Strong scaling**: Fix prompt set and batch size; compare 1 node (no pipeline) vs 2 nodes (pipeline size 2). Measure throughput, latency, and parallel efficiency.
- **Weak scaling**: Increase prompt count proportionally with node count (e.g., 10 prompts on 1 node → 20 prompts on 2 nodes).
- **Batch size sweep**: Evaluate batch sizes {1, 2, 4} to observe memory headroom and throughput changes under pipeline parallelism.
- **Quantization/optimization (optional)**: Explore BF16 baseline vs any lightweight quantization strategy if memory permits; note that current code path assumes BF16 weights.
- **Profiling hooks**: `slurm/run.sh` supports `PROFILER=nsys|perf|none`; currently defaults to none because the base image may lack nsys/perf. Nsight traces would show pipeline overlap and NCCL traffic if enabled.

## 9. Metrics and Instrumentation Plan
- **Wall-clock & per-prompt latency**: Logged per prompt in `rank_<r>.log` with tokens generated and throughput (tokens/s).
- **Throughput**: Derived from prompt generation timing; additional aggregated throughput can be computed post-hoc from completion JSONL and logs.
- **Parallel efficiency**: Compare strong-scaling speedups against ideal (2× for two nodes).
- **Resource checks**: NCCL debug logs for connectivity; optional `sacct` integration via `maybe_log_sacct` (called by rank 1 at end of run) can capture CPU time and memory.
- **GPU visibility/debug**: `run.sh` prints CUDA-related env and uses torch backend selection (`nccl` vs `gloo`) to confirm GPU availability in logs.

## 10. Limitations and Challenges
- **Resource constraints**: Limited GPU count and memory drove design toward a 3B/7B-class model and two-stage pipeline only; no data parallelism is implemented.
- **Container size and storage**: Building a fully self-contained Apptainer image was abandoned due to space; instead, we rely on a shared PyTorch image and runtime pip installs without torch/CUDA wheels.
- **Driver binding issues**: Early runs failed to find `libnvidia-ml.so` inside the container; addressed by adding `--nv`/`--nvccli` and explicit driver lib binding/`LD_LIBRARY_PATH` settings in the submission script.
- **Runtime dependency bootstrapping**: Network-restricted environments required explicit CA bundle selection and careful pip flags; caching is isolated under `/tmp/workspace/.venv` to reuse installs across runs.
- **Model placement trade-offs**: Current device map places only the active layers on GPU and offloads unused layers to CPU per rank. This conserves GPU memory but still incurs host RAM use; fully GPU-resident loading of all components would require more device memory.

## 11. Results (to be filled after experiments)

< Add detailed tables, charts, and narrative once measurements are collected. >

## 12. Analysis and Discussion (to be filled after experiments)

< Discuss throughput/latency, scaling efficiency, bottlenecks (compute vs communication), and impact of batch size or quantization. >

## 13. Future Work
- Extend to larger models and more stages; experiment with tensor parallelism layered on pipeline parallelism.
- Build a slimmer custom Apptainer image with baked-in dependencies to remove runtime pip installs.
- Integrate automated profiling (Nsight Systems) and structured metrics export for easier postprocessing.
- Add fault tolerance and retry logic for cluster runs; automate job submission from a controller node or CI-like workflow.
- Investigate memory optimizations (activation checkpointing, more aggressive offload) and mixed-precision strategies beyond BF16.
