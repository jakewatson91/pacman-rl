#!/usr/bin/env bash#
#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH -C A30  
#SBATCH --mem 12g 
#SBATCH --job-name="Train pacman" 

source activate p4
python main.py --train_dqn --data_dir "pacmantest/" --model_name "rainbow50_clip_nscaling_final.pth" --epsilon_decay_steps 2500000 --prioritized_beta_increment 0.0005 --episodes 50000 --max_buffer_size 100000 --buffer_start 20000 --v_max 1000 --v_min 0 --batch_size 64 --epsilon_min 0.05 --update_target_net_freq 8000 --life_penalty 0 --learning_rate 0.0000625 --n_scaling