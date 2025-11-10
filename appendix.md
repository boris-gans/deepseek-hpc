### Parallelism Defs

**Data Parallel (DP)**

* Replicates the *entire* model on multiple GPU groups; each group serves different requests.
* **Effect:** Improves **throughput/concurrency**, not single-request latency.
* **When to use:** Once you’ve optimized a single replica (TP/PP), add DP to scale QPS.

**Tensor Parallel (TP)**

* Splits *within-layer* math (e.g., big matmuls, attention heads) across GPUs.
* **Effect:** Reduces per-GPU memory for weights/activations; speeds up single-request latency when GPUs are connected by **very fast intra-node links** (NVLink/NVSwitch).
* **Caveat:** TP has heavy, frequent cross-GPU comms → keep it **within one NVLink/NVSwitch island**.

**Pipeline Parallel (PP)**

* Slices the **layers** into stages placed on different GPUs/nodes.
* **Effect:** Lets you fit very large models by distributing layers; adds “pipeline bubbles” unless you feed multiple microbatches.
* **Rule of thumb:** Use **as little PP as needed** to fit memory; balance stages by FLOPs so none becomes the bottleneck.

**Sequence parallel / KV-cache sharding** (often bundled with TP)

* Splits certain activation/state tensors (esp. KV cache) to reduce per-GPU memory.
* **Effect:** Big memory saver at long context lengths with minimal latency/accuracy tradeoff.

<hr>

### Attention Hyperparameters

* **L** — number of transformer **layers**.
  **Use:** `L = 80` (Llama-70B–class)

* **H** — number of attention **heads per layer**.
  **Use:** `H = 64`

* **D** — **dimensions per head** (head size).
  **Use:** `D = 128`  → hidden size = `H × D = 64 × 128 = 8192`

* **Precision** — BF16/FP16 (2 bytes/element).
  **Use:** **BF16** (same 2-byte footprint as FP16, safer numerically)

* **KV cache per token (full model)**
  Each generated (or prompt) token stores its **K** and **V** for *every layer and head*:
  `KV_per_token_bytes = 2 (K,V) × L × H × D × bytes = 2 × 80 × 64 × 128 × 2 = 2,621,440 bytes ≈ 2.5 MiB`

> Rule of thumb for sharding:
>
> * **Weights per GPU** = total_weights / (TP × PP)
> * **KV per GPU** (batch=1) = total_KV / (TP × PP)
> * (PP owns a subset of layers; TP shards tensors in each owned layer. Net effect is a divide by `TP×PP`.)

For “other activations” (non-KV) during inference and runtime **overhead** (CUDA/NCCL contexts, cuBLAS workspaces, fragmentation, logits/output buffers), a simple, conservative budget is:
* **Other activations:** ~**2 GiB** per GPU
* **Overhead:** ~**6 GiB** per GPU

<hr>

## Inference-only accelerators

These keep outputs identical (or extremely close) while improving speed/memory:

1. **FlashAttention / fused attention kernels**

   * Reduces memory traffic and speeds attention. No accuracy change.

2. **Paged attention + KV-cache sharding**

   * Cuts fragmentation and per-GPU KV footprint, especially at long contexts.

3. **CUDA Graphs + fused MLP kernels**

   * Stabilize launch overheads; better GPU utilization.

4. **Continuous (dynamic) batching + request scheduling**

   * Batch tokens by current step across requests to increase GPU occupancy.
   * Doesn’t change answers; just smarter scheduling.

5. **Speculative decoding (safe variant)**

   * Use a smaller draft model; the large model **verifies** tokens.
   * If verification is strict, final outputs are **identical** to baseline, with big speedups at moderate batch sizes.

6. **Pinned memory, NUMA-aware CPU threads, large page / allocator tuning**

   * Shaves overhead around the hot path.


## What to avoid to maximize accuracy

* **Aggressive weight quantization (INT4)** → fastest, but accuracy impact noticeable.
* **KV-cache INT4** → bigger quality hit than FP8/FP16-mix.
* **CPU/NVMe offload of weights** → huge latency penalty; only as a last resort to “make it run.”