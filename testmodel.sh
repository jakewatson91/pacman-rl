#!/usr/bin/env bash

#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32

#SBATCH --gres=gpu:1

#SBATCH -C A30  
#SBATCH -t 24:00:00
#SBATCH --mem 12g 

#SBATCH --job-name="Test Model" 


source $(conda info --base)/etc/profile.d/conda.sh

conda activate testenv

python main.py --test_dqn --data_dir 'updatedparams/' --model_name "expandedmodel.pth" 