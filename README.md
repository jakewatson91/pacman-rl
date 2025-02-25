Deep Reinforcement Learning Pac-Man Agent (Rainbow DQN)

Overview

This project trains an AI agent to play Pac-Man using Deep Reinforcement Learning with the Rainbow DQN architecture. The model was trained on a GPU for 50,000 epochs, incorporating prioritized experience replay, n-step returns, and distributional Q-learning.

Additionally, reward shaping (log rewards) and risk scaling were experimented with to address the sparse reward bottleneck in later levels.

Features
	•	Rainbow DQN: Integrates multiple DQN advancements for improved training efficiency.
	•	GPU Training: Utilizes CUDA for accelerated learning.
	•	Experience Replay: Implements prioritized experience replay for efficient sample selection.
	•	N-Step Learning: Uses multi-step return bootstrapping to improve learning stability.
	•	Distributional Q-Learning: Models uncertainty in value estimation.
	•	Reward Shaping: Tested log-based rewards to encourage exploration.
	•	Risk Scaling: Adjusted risk dynamically to enhance decision-making in sparse reward situations.
	•	WandB Integration: Logs training metrics for visualization.

Installation

pip install -r requirements.txt

Training

To train the agent, run:

python train.py

Evaluation

To test a trained model:

python test.py --model_path saved_model.pth

Results
	•	Training curves are logged in WandB.
	•	Plots of rewards and losses are saved in the output directory.
	•	Reward shaping and risk scaling were tested to mitigate sparse rewards in later levels.

Files
	•	agent.py – Implements the Rainbow DQN agent.
	•	dqn_model.py – Defines the neural network architecture.
	•	train.py – Main script for training the agent.
	•	test.py – Script to evaluate the trained model.

References
	1.	Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
	2.	Van Hasselt, H., Guez, A., & Silver, D. (2015). Deep reinforcement learning with double Q-learning. arXiv:1509.06461
	3.	Wang, Z., et al. (2015). Dueling Network Architectures for Deep Reinforcement Learning. arXiv:1509.06461
	4.	Schaul, T., et al. (2015). Prioritized Experience Replay. arXiv:1511.05952
	5.	Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. DOI:10.1007/BF00115009
	6.	Fortunato, M., et al. (2018). Noisy Networks for Exploration. arXiv:1706.10295
	7.	Bellemare, M. G., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. arXiv:1707.06887
	8.	Dabney, W., et al. (2018). Distributional Reinforcement Learning with Quantile Regression. arXiv:1710.10044
	9.	Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. arXiv:1710.02298
	10.	Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
	11.	Wikipedia contributors. Kullback–Leibler divergence. Wikipedia
	12.	Wikipedia contributors. Wasserstein metric. Wikipedia
	13.	Wijaya, S., et al. (2023). LSTM Model-Based Reinforcement Learning for Nonlinear Mass Spring Damper System Control. DOI:10.1016/j.procs.2022.12.129

This project builds upon fundamental advancements in deep reinforcement learning, incorporating techniques from DQN, Double DQN, Dueling DQN, and Distributional RL to create a robust Pac-Man playing agent. 🚀
