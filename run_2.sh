#!/bin/bash
#SBATCH --job-name=eval_FULL
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/nld68820/LIMIT/output_2.out
#SBATCH --error=/home/nld68820/LIMIT/output_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load Python/3.12.3
module load CUDA/12.8.0 
source /home/nld68820/.venv/bin/activate
cd "/rwthfs/rz/cluster/home/nld68820/LIMIT"

python -u main_2.py
