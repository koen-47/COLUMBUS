#!/bin/bash
#SBATCH --job-name="Llava 34b (Lateral Thinking Visual Brainteasers)"
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A6000
#SBATCH --gres=gpu:1

module load cuda11.7/toolkit

source $HOME/.bashrc
conda activate

python /var/scratch/hkd800/scripts/main.py llava-1.6-34b 1
