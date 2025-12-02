#!/bin/bash
#SBATCH --job-name=llama_pipeline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-node
#SBATCH --output=/home/%u/scratch/group1/hpc-runs/%x-%j.out
#SBATCH --error=/home/%u/scratch/group1/hpc-runs/%x-%j.err
# Update the paths above if you change SCRATCH_ROOT.

# submit.sbatch
# Single pipeline-parallel launch with fixed config paths.
# Usage: sbatch submit.sbatch (or via slurm/launch_pipeline.sh)

set -euo pipefail

#############################################
# Paths and image to adjust for your cluster
#############################################

# Shared project root on host (keep /home usage narrow). Use scratch for runtime data.
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/${USER}/scratch/group1}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_ROOT}/pipeline_run}"
CODE_ROOT="${CODE_ROOT:-/home/${USER}/projects/distributed-inference}"

# Fixed experiment root (host-side) matches the mount point /workspace inside the container.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}}"

# Apptainer image: prefer the shared PyTorch SIF provided on the cluster.
# Override via APPAINTER_IMAGE if your cluster stores it elsewhere.
APPAINTER_IMAGE="${APPAINTER_IMAGE:-/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif}"
if [[ -z "${APPAINTER_IMAGE}" ]]; then
  echo "APPAINTER_IMAGE is not set. Export it before calling sbatch." >&2
  exit 1
fi
if [[ ! -f "${APPAINTER_IMAGE}" ]]; then
  echo "Apptainer image not found at ${APPAINTER_IMAGE}. Override APPAINTER_IMAGE to the provided SIF." >&2
  exit 1
fi

if [[ ! -d "${CODE_ROOT}" ]]; then
  echo "CODE_ROOT ${CODE_ROOT} does not exist. Point CODE_ROOT to the repo with slurm/run.sh." >&2
  exit 1
fi

# Profiling mode for run.sh: "none", "nsys", or "perf"
export PROFILER="${PROFILER:-nsys}"

export PROJECT_ROOT
export EXPERIMENT_ROOT
export SCRATCH_ROOT
export EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${EXPERIMENT_ROOT}/exp_config.json}"
export DS_CONFIG_PATH="${DS_CONFIG_PATH:-${EXPERIMENT_ROOT}/ds_config.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_ROOT}/hpc-runs"

echo "Job ${SLURM_JOB_NAME} starting on nodes: ${SLURM_JOB_NODELIST}"
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Config: ${EXP_CONFIG_PATH}"
echo "DeepSpeed config: ${DS_CONFIG_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Image: ${APPAINTER_IMAGE}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Code root: ${CODE_ROOT}"

#############################################
# Launch containerized run.sh on all tasks
#############################################

# srun will start one task per node (because of --ntasks-per-node=1)
# Each task runs a separate Apptainer container, calling /app/slurm/run.sh inside.
# All tasks see the same /workspace (mounted from PROJECT_ROOT).

srun apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${CODE_ROOT}:/app" \
    "${APPAINTER_IMAGE}" \
    bash /app/slurm/run.sh

echo "Job ${SLURM_JOB_ID} finished."

echo "You can inspect accounting data with:"
echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,JobName%30,State,Elapsed,MaxRSS"
