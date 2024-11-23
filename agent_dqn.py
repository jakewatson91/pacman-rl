#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from noisy_linear import NoisyLinear
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.nn.utils as nn_utils

import wandb

import warnings
warnings.filterwarnings("ignore")

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        # start a new wandb run to track this script
        wandb.init(project="CS525-MsPacman")
        self.config = wandb.config
        self.config.update(vars(args))

        super(Agent_DQN,self).__init__(env)
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #main
        self.episodes = args.episodes
        self.update_target_net_freq = args.update_target_net_freq
        self.greedy_steps = 0
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_max = args.epsilon
        self.epsilon_decay_steps = args.epsilon_decay_steps

        #prio   
        self.prioritized_alpha = args.prioritized_alpha
        self.beta = args.prioritized_beta
        self.beta_increment = args.prioritized_beta_increment

        #distributional
        from dist_config import DistributionalConfig
        self.dist_config = DistributionalConfig(args.num_atoms, args.v_max, args.v_min, self.device)
        self.num_atoms = self.dist_config.num_atoms
        self.v_min = self.dist_config.v_min
        self.v_max = self.dist_config.v_max
        self.delta_z = self.dist_config.delta_z  # Atom step size
        self.support = self.dist_config.support  # Atom values

        #multi-step
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        #configure data directories and logging
        self.data_dir = args.data_dir
        self.model_name = args.model_name
        self.save_freq = args.save_freq
        self.write_freq = args.write_freq
        self.print_freq = args.print_freq

        # initialize buffers
        self.max_buffer_size = args.max_buffer_size
        self.buffer_start = args.buffer_start
        self.buffer = deque(maxlen=self.max_buffer_size)
        self.priorities = deque(maxlen=self.max_buffer_size)

        #init model and target net
        self.q_net = DQN(dist_config=self.dist_config).to(self.device)
        self.target_net = DQN(dist_config=self.dist_config).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())     
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.learning_rate)

        if args.test_dqn or args.train_dqn_again:
            print('loading trained model')

            checkpoint = torch.load(self.data_dir + self.model_name, weights_only=True)
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])

            self.epsilon = self.epsilon_min                      

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass 
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """

        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        state = state.permute(0, 3, 1, 2)

        if test:
            self.greedy_steps += 1
            epsilon = 0
        else:
            epsilon = self.epsilon

        action = self.q_net.act(state, epsilon)
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """

        # Store transitions in an intermediate buffer for n-step returns
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Compute the n-step reward
            state,action,n_step_reward, n_step_next_state, n_step_done = self.get_n_step_info()

            # Take the first transition from the n-step buffer
            state, action, _, _, _ = self.n_step_buffer[0]

            # Append the n-step transition to the main buffer
            self.buffer.append((state, action, n_step_reward, n_step_next_state, n_step_done))

            # Set priority for the new transition
            priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority)
    
    def get_n_step_info(self):
        """
        Get the n-step info
        """
        if len(self.n_step_buffer) < self.n_step:
            return None

        state, action,reward, next_state, done = self.n_step_buffer[-1]

        for i in range(self.n_step-2, -1, -1):
            s,a,r, ns, d = self.n_step_buffer[i]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (ns, d) if d else (next_state, done)

        state, action = self.n_step_buffer[0][:2]

        return state, action, reward, next_state, done
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """

        batch = random.sample(self.buffer, self.batch_size)

        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.int64).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(done), dtype=torch.float32).to(self.device)

        return state, action, reward, next_state, done         

    def prioritized_replay_buffer(self):
        '''
        Calculate the priority of each transition and sample from the buffer based on the priority
        
        '''
        
        priorities = np.array(self.priorities) ** self.prioritized_alpha
        priorities /= priorities.sum()

        batch = random.choices(self.buffer, weights=priorities, k=self.batch_size)
        self.beta = min(1.0, self.beta + self.beta_increment)

        weights = (len(self.buffer) * priorities) ** -self.beta
        weights /= weights.max()

        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.int64).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(done), dtype=torch.float32).to(self.device)

        return state, action, reward, next_state, done, weights
    
    def project_distribution(self, reward, next_probabilities, done): #for distributional
        batch_size = reward.size(0)
        projected_distribution = torch.zeros((batch_size, self.num_atoms), device=self.device)

        for j in range(self.num_atoms):
            # Bellman update
            tz_j = reward + (1 - done) * (self.gamma * (self.v_min + j * self.delta_z))
            tz_j = tz_j.clamp(self.v_min, self.v_max)

            # Calculate projection
            b = (tz_j - self.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()

            # Initialize the projected distribution with zeros
            projected_distribution = torch.zeros_like(next_probabilities, device=next_probabilities.device)

            # Compute the contributions for lower and upper bins
            lower_contrib = next_probabilities * (u - b).unsqueeze(1)
            upper_contrib = next_probabilities * (b - l).unsqueeze(1)

            # Expand l and u to match the dimensions of the contributions
            l_expanded = l.unsqueeze(1).expand(-1, self.num_atoms)
            u_expanded = u.unsqueeze(1).expand(-1, self.num_atoms)

            # print("projected_distribution shape:", projected_distribution.shape)  # Expected: [batch_size, num_atoms]
            # print("l_expanded shape:", l_expanded.shape)  # Expected: [batch_size, num_atoms]
            # print("lower_contrib shape:", lower_contrib.shape)  # Expected: [batch_size, num_atoms]

            # Scatter the contributions to their respective bins
            projected_distribution.scatter_add_(1, l_expanded, lower_contrib)
            projected_distribution.scatter_add_(1, u_expanded, upper_contrib)

        return projected_distribution

    def fill_buffer(self):
        state = self.env.reset()

        while len(self.buffer) < self.buffer_start:
            #get action from q_net
            action = self.make_action(state, test=False)
            next_state, reward, done, _,_ = self.env.step(action)
            self.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env.reset()
        print('Buffer filled')

    def train(self):

        """
        Implement your training algorithm here
        """
        
        all_rewards = []
        avg_rewards = []

        self.losses = []
        epsilons = [self.epsilon]

        self.fill_buffer()
        
        # episode =0

        for episode in tqdm(range(self.episodes)): 

            # if episode>0 and episode % self.print_freq == 0:
            #     print(f'Episode: {episode}, Reward: {all_rewards[-1]},Avg Reward: {avg_rewards[-1]}')

            state = self.env.reset()
            done = False
            episode_reward = 0

            self.epsilon = self.update_epsilon(episode)

            epsilons.append(self.epsilon)    
            # run episode
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _,_ = self.env.step(action)
                self.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            self.train_batch()
            all_rewards.append(episode_reward)
            avg_rewards.append(np.mean(all_rewards[-30:]))
  
            if episode % self.update_target_net_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % self.write_freq == 0:
                #plot avg rewards
                self.makePlots(all_rewards, avg_rewards, self.losses,epsilons)

            if episode > 0 and episode % self.save_freq == 0:

                # checkpoint model, if it is the best model so far save it

                torch.save({
                    'model_state_dict': self.q_net.state_dict(),
                }, self.data_dir + self.model_name)

                print('Model saved')
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
        }, self.data_dir + self.model_name)

        wandb.finish() #close wandb job
        print('Training Complete')

    def train_batch(self):
        state, action, reward, next_state, done, weights = self.prioritized_replay_buffer()

        state = state.permute(0, 3, 1, 2)
        next_state = next_state.permute(0, 3, 1, 2)
        # print("state shape: ", state.shape)
        # print("next state shape: ", next_state.shape)

        # Get the distributions from the networks -- distributional DQN
        probabilities = self.q_net(state).to(self.device)
        # print("probs shape: ", probabilities.shape)
        next_probabilities = self.target_net(next_state).to(self.device)
        # print("next probs shape: ", next_probabilities.shape)
        # print("support shape: ", self.support.shape)
        # print("sum shape: ", torch.sum(next_probabilities * self.support, dim=2).shape)

        # Compute the target distribution
        next_q_values = torch.sum(next_probabilities * self.support, dim=2)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Gather the target probabilities for the chosen actions
        next_probabilities = next_probabilities[range(self.batch_size), next_actions]
        
        target_distribution = self.project_distribution(reward, next_probabilities, done)

        # Compute loss (KL divergence)
        log_probabilities = torch.log(probabilities[range(self.batch_size), action])
        loss = F.kl_div(log_probabilities, target_distribution, reduction='batchmean')

        # commented out original
        # q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # q_values_next = self.target_net(next_state).gather(1, self.q_net(next_state).argmax(1).unsqueeze(1)).squeeze(1).detach() # double DQN

        # q_target = reward + (self.gamma ** self.n_step) * q_values_next * (1 - done)

        # loss = F.huber_loss(q_values, q_target).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()

        nn_utils.clip_grad_norm_(self.q_net.parameters(), 1)

        self.optimizer.step()
        self.q_net.reset() #reset noise after every forward pass

        self.losses.append(loss.item())

        self.priorities = deque(weights, maxlen=self.max_buffer_size)

    def makePlots(self, all_rewards, avg_rewards, losses, epsilons):
        plt.plot(avg_rewards, color='red')
        #add thick line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.title('Avg Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Avg Rewards')
        
        plt.savefig(self.data_dir+'avg_rewards.png')
        wandb.log({"Avg Rewards Plot": wandb.Image(self.data_dir + 'avg_rewards.png')})
        plt.clf()


        #plot rewards
        plt.plot(all_rewards)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

        plt.title('Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(self.data_dir+'rewards.png')
        wandb.log({"Rewards Plot": wandb.Image(self.data_dir + 'rewards.png')})
        plt.clf()

        plt.plot(np.log(losses))
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

        plt.title('Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.savefig(self.data_dir+'loss.png')
        wandb.log({"Loss Plot": wandb.Image(self.data_dir + 'loss.png')})
        plt.clf()

        plt.close()

            # Log raw data to W&B
        for i, (reward, avg_reward, loss, epsilon) in enumerate(zip(all_rewards, avg_rewards, losses, epsilons)):
            wandb.log({
                "Episode": i + 1,
                "Reward": reward,
                "Avg Reward": avg_reward,
                "Loss": loss,
                "Epsilon": epsilon
            })

        # Write to file
        np.savetxt(self.data_dir+'/rewards.csv', all_rewards, delimiter=',', fmt='%d')
        np.savetxt(self.data_dir+'/avg_rewards.csv', avg_rewards, delimiter=',', )
        np.savetxt(self.data_dir+'/loss.csv', losses, delimiter=',', )
        np.savetxt(self.data_dir+'/epsilon.csv', epsilons, delimiter=',', )


    def update_epsilon(self, episode):
        self.epsilon -= (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon