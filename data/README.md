<!-- Data management notes covering storage, preprocessing, and retention for DeepSeek evaluations. -->

We want to measure the quality of outputs as well, meaning we measure:

1. Deterministic equivalence (primary): With identical decoding settings (greedy or temperature=0), the distributed setup should match a baseline DeepSeek v3.1 run.

2. Task realism (secondary): On a tiny curated set (10–50 prompts), outputs should be comparable to a baseline judged by simple automatic metrics or pairwise preference.


In our case, quality can change due to two main reasons. We want to detect this to ensure our model behaves properly:

1. Precision (FP16 vs BF16), kernel choices, tensor/pipeline parallel collectives, and quantization can slightly perturb logits → different tokens with sampling.

2. Different tokenizers, max lengths, or server-side system prompts also cause drift.


## Deterministic Correctness Checks


1. Logit match test on 20 fixed prompts: compare per-token logits against the baseline run, DeepSeek v3.1 hosted on DeepSeek servers
    - Using the same hyperparameters: Greedy decoding (temperature=0, top_p=1, top_k=0, do_sample=False), dropout disabled, fixed seeds, identical tokenizer

    - Report mean absolute error and max absolute error.

2. Batch-size invariance: same prompts run as batch=1 vs batch=8 should produce identical outputs under greedy.

3. Multi-node invariance: 1-node vs 2-node sharding yields identical outputs.