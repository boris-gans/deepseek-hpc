# llama_pipeline-4205.err
srun: error: gpu-node1: task 0: Exited with exit code 1
srun: error: gpu-node2: task 1: Exited with exit code 1

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
[rank 1] Running without profiler...
[rank 0] Running without profiler...
Traceback (most recent call last):
  File "/app/run_distributed_inference.py", line 30, in <module>
    import deepspeed
ModuleNotFoundError: No module named 'deepspeed'
Traceback (most recent call last):
  File "/app/run_distributed_inference.py", line 30, in <module>
    import deepspeed
ModuleNotFoundError: No module named 'deepspeed'


# what to measure
Wall-clock time, throughput (e.g., it/s, steps/s), parallel efficiency (% of ideal).
Resource use (GPU/CPU utilization, memory BW, PCIe/NVLink traffic if available).
I/O time (read/write), communication time (MPI, NCCL).
Cost of data-loader / preprocessing for AI workloads.