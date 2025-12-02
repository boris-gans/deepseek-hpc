## Run with the shared PyTorch image (no build)

1) Sync the tiny code folder to `/home` (or your project repo path) on the cluster:
```bash
rsync -av \
  slurm \
  run_distributed_inference.py \
  user49@hpcie.labs.faculty.ie.edu:/home/user49/projects/def-sponsor00/user49/distributed-inference
```

2) Prepare the shared scratch workspace that all ranks will see as `/workspace`:
```bash
ssh user49@hpcie.labs.faculty.ie.edu <<'EOF'
  SCRATCH_ROOT=/home/user49/scratch/group1
  PIPELINE_ROOT=${SCRATCH_ROOT}/pipeline_run

  mkdir -p ${PIPELINE_ROOT}/outputs ${SCRATCH_ROOT}/hpc-runs
  # Place prompts + configs under scratch so every node reads the same files
  cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/exp_config.json ${PIPELINE_ROOT}/exp_config.json
  cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/ds_config.json  ${PIPELINE_ROOT}/ds_config.json
  cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/prompts.jsonl   ${PIPELINE_ROOT}/prompts.jsonl
EOF
```
*Prompts*: the job reads whatever path is in `exp_config.json["inputs"]["prompt_path"]`. The defaults expect `/workspace/prompts.jsonl`, which is the same file you copied into `${PIPELINE_ROOT}`.

3) Point to the shared PyTorch image and submit:
```bash
export APPAINTER_IMAGE=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
export PIPELINE_ROOT=/home/user49/scratch/group1/pipeline_run
export CODE_ROOT=/home/user49/projects/def-sponsor00/user49/distributed-inference
sbatch ${CODE_ROOT}/slurm/submit.sbatch
```
The sbatch script will bind `${PIPELINE_ROOT} -> /workspace` (for prompts/configs/outputs) and `${CODE_ROOT} -> /app` (for `run.sh` + `run_distributed_inference.py`).

4) After the job, check logs/outputs under `${PIPELINE_ROOT}/outputs` and the Slurm stdout/err under `${SCRATCH_ROOT}/hpc-runs`.

---

## Appendix (legacy image build / sync commands)
These are kept for reference if you ever need to build your own SIF; not required when using the shared PyTorch image.

Transport files:
```bash
rsync -av env/apptainer.def user49@hpcie.labs.faculty.ie.edu:~/distributed-inference/env

rsync -av \
  slurm \
  run_distributed_inference.py \
  env/apptainer.def \
  user49@login1:/home/user49/distributed-inference/env
```

Build on scratch to avoid quotas:
```bash
SCRATCH_ROOT=/home/user49/scratch/group1
mkdir -p ${SCRATCH_ROOT}/appainter

export APPTAINER_TMPDIR=/home/user49/scratch/group1/tmp
export APPTAINER_CACHEDIR=/home/user49/scratch/group1/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

apptainer build \
    ${SCRATCH_ROOT}/appainter/appainter.sif \
    /home/user49/distributed-inference/env/apptainer.def

apptainer exec \
  --mount type=image,source=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif,target=/ext/pytorch \
  --mount type=image,source=/home/user49/projects/def-sponsor00/shared/images/gromacs.sif,target=/ext/gromacs \
  ${SCRATCH_ROOT}/appainter/appainter.sif \
  python3 /app/run_distributed_inference.py
```

Copy a prebuilt SIF into scratch (if you built elsewhere):
```bash
rsync -av ${SCRATCH_ROOT}/appainter/appainter.sif user49@login1:${SCRATCH_ROOT}/appainter/
```
