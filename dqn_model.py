#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from noisy_linear import NoisyLinear

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, dist_config, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_actions = num_actions
        self.num_atoms = dist_config.num_atoms
        self.support = dist_config.support
        
        #main network
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        #dueling networks
        value_layer_1 = NoisyLinear(64 * 7 * 7, 512) #for use in reset
        value_layer_2 = NoisyLinear(512, 1 * self.num_atoms)
        
        self.fc_value = nn.Sequential(
            value_layer_1,
            nn.ReLU(),
            value_layer_2,

        )

        adv_layer_1 = NoisyLinear(64 * 7 * 7, 512) #for use in reset
        adv_layer_2 = NoisyLinear(512, num_actions * self.num_atoms)
        
        self.fc_advantage = nn.Sequential(
            adv_layer_1,
            nn.ReLU(),
            adv_layer_2,
        )

        self.layers = [value_layer_1, value_layer_2, adv_layer_1, adv_layer_2]

    def forward(self, x):
        # if isinstance(x
        x = self.features(x)
        x = x.reshape(x.size(0), -1)

        value_logits = self.fc_value(x)
        # print("value logits shape: ",value_logits.shape)

        value_logits = value_logits.view(-1, 1, self.num_atoms)  # Shape: [batch_size, 1, num_atoms]
        # print("value logits shape: ",value_logits.shape)
        advantage_logits = self.fc_advantage(x).view(-1, self.num_actions, self.num_atoms)  # Shape: [batch_size, num_actions, num_atoms]
    
        # Combine value and advantage logits
        logits = value_logits + (advantage_logits - advantage_logits.mean(dim=1, keepdim=True))  # Broadcasting
        # print("logits shape: ", logits.shape)
        # print("logits: ", logits)
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=2)  # Softmax along atom dimension

        return probabilities
    
    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        state = torch.as_tensor(state).to(self.device)
        probabilities = self.forward(state)  # Get action distributions
        q_values = torch.sum(probabilities * self.support, dim=2)
        return torch.argmax(q_values, dim=1).item()
    
    def reset(self):
        for layer in self.layers:
            layer.reset_noise()
