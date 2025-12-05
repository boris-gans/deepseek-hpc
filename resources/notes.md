# llama_pipeline-4205.err
Lmod has detected the following error: These module(s) or extension(s) exist
but cannot be loaded as requested: "apptainer"
   Try: "module spider apptainer" to see how to load the module(s).



Lmod has detected the following error: These module(s) or extension(s) exist
but cannot be loaded as requested: "cuda"
   Try: "module spider cuda" to see how to load the module(s).



srun: error: gpu-node2: task 1: Exited with exit code 1
srun: error: gpu-node1: task 0: Exited with exit code 1

# llama_pipeline-4205.out
Job llama_pipeline starting on nodes: gpu-node[1-2]
Experiment root: /tmp/workspace
Config: /tmp/workspace/exp_config.json
DeepSpeed config: /tmp/workspace/ds_config.json
Output dir: /tmp/workspace/outputs
Image: /home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
Scratch root: /home/user49/scratch/group1
Code root: /home/user49/projects/def-sponsor00/user49/distributed-inference
Container workspace: /tmp/workspace
Master addr: gpu-node1
[rank 0] Using CA bundle at /opt/conda/lib/python3.10/site-packages/certifi/cacert.pem
[rank 1] Using CA bundle at /opt/conda/lib/python3.10/site-packages/certifi/cacert.pem
[rank 0] Running without profiler...
[rank 1] Running without profiler...
[W Utils.hpp:135] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
[W Utils.hpp:135] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
2025-12-05 10:30:11,818 [INFO][rank=1] Initialized torch.distributed (rank=1, world_size=2)
2025-12-05 10:30:11,819 [INFO][rank=1] Using Hugging Face cache at /tmp/workspace/hf_cache
2025-12-05 10:30:11,822 [INFO][rank=0] Initialized torch.distributed (rank=0, world_size=2)
2025-12-05 10:30:11,823 [INFO][rank=0] Using Hugging Face cache at /tmp/workspace/hf_cache
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message
2025-12-05 10:30:11,920 [INFO][rank=1] Loaded tokenizer for /workspace/models/openllama-3b
2025-12-05 10:30:11,920 [INFO][rank=1] Loading model /workspace/models/openllama-3b for pipeline stage 1...
2025-12-05 10:30:11,935 [INFO][rank=0] Loaded tokenizer for /workspace/models/openllama-3b
2025-12-05 10:30:11,935 [INFO][rank=0] Loading model /workspace/models/openllama-3b for pipeline stage 0...
2025-12-05 10:30:11,934 [INFO][rank=1] Stage 1 device_map={'model.layers.13': 'cuda', 'model.layers.14': 'cuda', 'model.layers.15': 'cuda', 'model.layers.16': 'cuda', 'model.layers.17': 'cuda', 'model.layers.18': 'cuda', 'model.layers.19': 'cuda', 'model.layers.20': 'cuda', 'model.layers.21': 'cuda', 'model.layers.22': 'cuda', 'model.layers.23': 'cuda', 'model.layers.24': 'cuda', 'model.layers.25': 'cuda', 'model.norm': 'cuda', 'lm_head': 'cuda', 'model.embed_tokens': 'cpu'}
2025-12-05 10:30:11,952 [INFO][rank=0] Stage 0 device_map={'model.layers.0': 'cuda', 'model.layers.1': 'cuda', 'model.layers.2': 'cuda', 'model.layers.3': 'cuda', 'model.layers.4': 'cuda', 'model.layers.5': 'cuda', 'model.layers.6': 'cuda', 'model.layers.7': 'cuda', 'model.layers.8': 'cuda', 'model.layers.9': 'cuda', 'model.layers.10': 'cuda', 'model.layers.11': 'cuda', 'model.layers.12': 'cuda', 'model.s': 'cuda', 'model.norm': 'cpu', 'lm_head': 'cpu'}
[rank1]: Traceback (most recent call last):
[rank1]:   File "/app/run_distributed_inference.py", line 484, in <module>
[rank1]:     main()
[rank1]:   File "/app/run_distributed_inference.py", line 451, in main
[rank1]:     stage_module, hidden_size = partition_model(
[rank1]:   File "/app/run_distributed_inference.py", line 175, in partition_model
[rank1]:     model = AutoModelForCausalLM.from_pretrained(
[rank1]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 564, in from_pretrained
[rank1]:     return model_class.from_pretrained(
[rank1]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3916, in from_pretrained
[rank1]:     ) = cls._load_pretrained_model(
[rank1]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 4390, in _load_pretrained_model
[rank1]:     new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
[rank1]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 918, in _load_state_dict_into_meta_model
[rank1]:     raise ValueError(f"{param_name} doesn't have any device set.")
[rank1]: ValueError: model.layers.0.self_attn.q_proj.weight doesn't have any device set.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/run_distributed_inference.py", line 484, in <module>
[rank0]:     main()
[rank0]:   File "/app/run_distributed_inference.py", line 451, in main
[rank0]:     stage_module, hidden_size = partition_model(
[rank0]:   File "/app/run_distributed_inference.py", line 175, in partition_model
[rank0]:     model = AutoModelForCausalLM.from_pretrained(
[rank0]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 564, in from_pretrained
[rank0]:     return model_class.from_pretrained(
[rank0]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3916, in from_pretrained
[rank0]:     ) = cls._load_pretrained_model(
[rank0]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 4390, in _load_pretrained_model
[rank0]:     new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
[rank0]:   File "/tmp/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 918, in _load_state_dict_into_meta_model
[rank0]:     raise ValueError(f"{param_name} doesn't have any device set.")
[rank0]: ValueError: model.embed_tokens.weight doesn't have any device set.