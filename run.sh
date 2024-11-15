#!/usr/bin/env bash

#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32

#SBATCH --gres=gpu:1

#SBATCH -C A30  
#SBATCH --mem 12g 


#SBATCH --job-name="Train pacman" 


module load miniconda3

module load cuda 

source $(conda info --base)/etc/profile.d/conda.sh

conda activate testenv

# pip install opencv-python-headless gymnasium[atari] autorom[accept-rom-license]
# pip install -U "ray[rllib]" ipywidgets

# conda install -c conda-forge tqdm

# conda install -c conda-forge moviepy ffmpeg
# conda update ffmpeg

# source myenv/bin/activate
python main.py --train_dqn --data_dir "pacmantest/" --model_name "halfrainbow.pth"

#writes to slurm, or in python can write to a file 1 or 2 gpus, up to 4, cpus up to 96, change gres line up to 4, no need for explicit time constraitn max 2 days

# for jupyter