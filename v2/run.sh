#!/usr/bin/env bash
#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH -C A30  
#SBATCH --mem 12g 
#SBATCH --job-name="Train pacman" 

source activate p4
python main.py --train_dqn --data_dir "pacmantest/" --model_name "rainbow20_clip10_2.pth" --epsilon_decay_steps 1000000 --prioritized_beta_increment 0.0001 --n_step 5 --episodes 20000 --max_buffer_size 100000 --buffer_start 20000 --v_max 10 --v_min -100 --batch_size 64