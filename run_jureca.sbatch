#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --error=mpi-err.%j
#SBATCH --output=mpi-out.%j
#SBATCH --partition=dc-gpu-devel
#SBATCH --account=jicg41
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=v.sivaprasad@fz-juelich.de
#SBATCH --mail-type=END



module load Stages/2024 GCCcore/.12.3.0 shpc/0.1.26
module load nvcr.io/nvidia/tensorflow/23.10-tf2-py3/module



export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_GPU_ALLOCATOR=cuda_malloc_async
# train
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" tensorflow-run /p/scratch/share/sivaprasad1/visakh/launch.sh