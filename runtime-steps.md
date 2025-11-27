1) Build the Apptainer image locally (or on the cluster login node if allowed):
```bash
apptainer build env/appainter.sif env/apptainer.def
```

2) Push only the necessary artifacts to the cluster (`/home` for code only):
```bash
rsync -av \
  slurm \
  run_distributed_inference.py \
  env/appainter.sif \
  user49@login1:~/projects/def-sponsor00/shared/group1
```

3) On the cluster, place artifacts on the correct filesystems:
```bash
# Persist the container image on /project
ssh user49@login1 <<'EOF'
  mkdir -p /project/user49/appainter
  mv ~/projects/distributed-inference/env/appainter.sif /project/user49/appainter/appainter.sif

  # Prepare shared runtime root on /scratch (mounted to /workspace in the job)
  mkdir -p /scratch/user49/pipeline_run/outputs
  # Copy or create your configs and prompts under /scratch so they are shared:
  # cp ~/projects/distributed-inference/exp_config.json /scratch/user49/pipeline_run/exp_config.json
  # cp ~/projects/distributed-inference/ds_config.json /scratch/user49/pipeline_run/ds_config.json
  # cp ~/projects/distributed-inference/prompts.jsonl  /scratch/user49/pipeline_run/prompts.jsonl
EOF
```

4) Submit from the login node using the updated defaults (scratch mount and project image):
```bash
export APPAINTER_IMAGE=/project/user49/appainter/appainter.sif
export PROJECT_ROOT=/scratch/user49/pipeline_run
sbatch ~/projects/distributed-inference/slurm/submit.sbatch
```
