#!/bin/bash
#SBATCH --job-name=eval_FULL
#SBATCH --partition=c23ms
#SBATCH --time=03:00:00
#SBATCH --output=/home/nld68820/LIMIT/output_2.out
#SBATCH --error=/home/nld68820/LIMIT/output_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load Python/3.12.3
source /home/nld68820/.venv/bin/activate

python main_2.py
