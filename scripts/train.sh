#!/bin/bash
#SBATCH --job-name=seeing-unseen
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 4
#SBATCH --signal=USR1@1000
#SBATCH --partition=long,short
#SBATCH --constraint="a40"
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --requeue

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR
export DEFAULT_PORT=8739

export NGPUS=$(nvidia-smi --list-gpus | wc -l)

echo "num gpus: ${NGPUS}"

srun python seeing_unseen/run.py config/baseline/clip_unet.yaml \
  run_type=train \
  training.epochs=25 \
  training.batch_size=32 \
  training.lr=0.0003 \
  dataset.root_dir="data/datasets/semantic_placement" \
  checkpoint_dir="/path/to/checkpoint/dir/"
