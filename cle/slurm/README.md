<!-- Slurm usage notes for submitting, monitoring, and interpreting distributed inference jobs. -->

### Submit.sbatch

The Slurm submission script. It:
-   defines job resources
-   launches the appainter container

IUt mounts a shared host directory inside the container, containing:
-   exp_config.json
-   ds_config.json
-   input files
-   an outputs directory

### run.sh

The script that gets executed inside the container. It:
-   Reads env variables set by Slurm (above)
-   Launch the actual distributed job

### launch_pipeline.sh (helper)

Just a front-end wrapper. It:
-   ensures the env variable APPAINTER_IMAGE is set
-   checks whether the shared host directory contains the required files
-   creates missing items automatically (empty configs or directories)

It calls **submit.sbatch**