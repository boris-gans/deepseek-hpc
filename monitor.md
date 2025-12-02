```bash
squeue -j <jobid>

scontrol show job <jobid>

sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocCPUS,AllocGRES

watch -n 1 squeue -u $USER


```


export APPAINTER_IMAGE=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif