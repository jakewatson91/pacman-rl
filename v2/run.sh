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
python main.py --train_dqn --data_dir "pacmantest/" --model_name "rainbow10_final.pth" --epsilon_decay_steps 150000 --prioritized_beta_increment 0.001 --episodes 10000 --max_buffer_size 100000 --buffer_start 20000 --v_max 2.5 --v_min -10 --batch_size 64 --epsilon_min 0.05 --update_target_net_freq 8000 --life_penalty 10 --n_scaling