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
from torch import amp  # Updated import for AMP

import matplotlib.pyplot as plt

from agent import Agent
from dqn_model import DQN
# from plot import Plot

import wandb

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

        super(Agent_DQN,self).__init__(env)

        wandb.init(project="CS525-MsPacman")
        # print("wandb mode: ", wandb.run.settings)
        self.config = wandb.config
        self.config.update(vars(args))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.loss = torch.nn.HuberLoss()

        self.steps = 0
        
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

        self.rewards = []
        self.losses = []

        #configure data directories and logging
        self.data_dir = args.data_dir
        self.model_name = args.model_name
        self.save_freq = args.save_freq
        self.write_freq = args.write_freq
        self.print_freq = args.print_freq

        # initialize buffers
        self.max_buffer_size = args.max_buffer_size
        self.buffer_start = args.buffer_start
        self.buffer = deque(maxlen=10000) #changing to list for efficiency
        self.priorities = []

        #init model and target net
        self.q_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())     
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.learning_rate)

        if args.test_dqn or args.train_dqn_again:
            print('loading trained model')

            checkpoint = torch.load(self.data_dir + self.model_name, weights_only=True)
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])

            self.epsilon = self.epsilon_min

        # Enable CUDNN Benchmarking for optimized convolution operations
        torch.backends.cudnn.benchmark = True
            
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
     
    def make_action(self, observation, num_actions=4, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        # print("Obs tensor: ", obs_tensor)
        
        with amp.autocast(device_type=self.device.type):
            q_vals = self.q_net(obs_tensor)
            # print("q_vals: ", q_vals)
        
        # print("epsilon: ", self.epsilon)
        if not test and random.random() < self.epsilon:
            # print("Action space: ", self.env.action_space.n)
            action = self.env.action_space.sample()
            # print("Random action from make_action: ", action)
        else:
            action = torch.argmax(q_vals, dim=1).item()
            # print("Max action from make_action: ", action)

        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append((state, action, reward, next_state, done)) 
           
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        batch = random.sample(self.buffer, k=self.batch_size)
        return batch 

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

    def update(self): 
        experiences = self.replay_buffer()
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0,3,1,2).to(self.device, non_blocking=True)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device, non_blocking=True)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device, non_blocking=True)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0,3,1,2).to(self.device, non_blocking=True)
        dones = torch.tensor(np.array(dones), dtype=torch.int64).to(self.device, non_blocking=True)

        # # Validate tensor shapes
        # assert actions.shape[0] == self.batch_size, "Invalid batch size for actions"
        # assert (actions >= 0).all() and (actions < self.q_net(states).shape[1]).all(), "Invalid action indices"

        #double implementation
        qs = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) #compute states and next states on main network
        next_qs = self.q_net(next_states)
        
        best_actions = torch.argmax(next_qs, dim=1) #get action indices for best state transition

        target_qs = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1) #compute targets using action indices from main network
        targets = rewards + (1 - dones) * self.gamma * target_qs

        loss = self.loss(qs, targets.detach()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def train(self):
        """
        Implement your training algorithm here
        """
        all_rewards = []
        avg_rewards = []
        self.losses = []
        epsilons = [self.epsilon]

        self.fill_buffer()

        for episode in tqdm(range(self.episodes)):
            state = self.env.reset()
            # print("initial state shape: ", state.shape)
            done = False

            total_reward = 0
            episode_loss = 0
            steps = 0

            while not done:
                loss = None
                action = self.make_action(state, test=False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.push(state, action, reward, next_state, done) # push to replay buffer

                total_reward += reward 
                self.steps += 1 # global steps
                steps += 1 # episode steps

                if self.steps % 4 == 0:
                    loss = self.update()
                if loss is not None:
                    episode_loss += loss.item()

                if self.steps > 10000 and self.epsilon > self.epsilon_min:
                    self.update_epsilon(episode)
                    # if self.steps % 10 == 0:
                    #     print("Epsilon: ", self.epsilon)
                
                if self.steps % 5000 == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                state = next_state
            
            all_rewards.append(total_reward)
            avg_rewards.append(np.mean(all_rewards[-30:]))
            self.losses.append(episode_loss)
            epsilons.append(self.epsilon)
            
            # plotting, logging, saving
            if episode and episode % self.write_freq == 0:
                
                self.makePlots(all_rewards, avg_rewards, self.losses, epsilons, _, _)

                logger.info(f"Episode {episode+1}: Loss = {episode_loss}")
                logger.info(f"Episode {episode+1}: Avg Rewards = {avg_rewards[-1]}")
                logger.info(f"Episode {episode+1}: Epsilon = {self.epsilon}")
                logger.info(f"Episode {episode+1}: Steps this episode = {steps}")

                torch.save({
                    'model_state_dict': self.q_net.state_dict(),
                }, self.data_dir + self.model_name)
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
        }, self.data_dir + self.model_name)

        wandb.finish()

    def makePlots(self, all_rewards, avg_rewards, losses, epsilons, exploration_steps, exploitation_steps):
        plt.plot(avg_rewards, color='red')
        #add thick line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Avg Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Avg Rewards')
        
        plt.savefig(self.data_dir+'avg_rewards.png')
        wandb.log({"Avg Rewards Plot": wandb.Image(self.data_dir + 'avg_rewards.png')})
        plt.clf()


        #plot rewards
        plt.plot(all_rewards)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.title('Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(self.data_dir+'rewards.png')
        wandb.log({"Rewards Plot": wandb.Image(self.data_dir + 'rewards.png')})
        plt.clf()

        plt.plot(np.log(losses))
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.title('Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.savefig(self.data_dir+'loss.png')
        wandb.log({"Loss Plot": wandb.Image(self.data_dir + 'loss.png')})
        plt.clf()

        # #make sure agent is properly exploring
        # plt.plot(exploration_steps, color='green')
        # plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # plt.plot(exploitation_steps, color='purple')
        # plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # plt.legend()
        # plt.title('Exploration/Exploitation')
        # plt.xlabel('Episodes')
        # plt.ylabel('Actions')
        # plt.savefig(self.data_dir+'exploration_exploitation.png')
        # wandb.log({"Exploration/Exploitation": wandb.Image(self.data_dir + 'exploration_exploitation.png')})
        
        # plt.clf()

        plt.close()

        #Log raw data to W&B
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