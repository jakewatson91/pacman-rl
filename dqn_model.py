#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, args, in_channels=4, num_actions=5):
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

        self.args = args
        self.num_actions = num_actions
        self.num_atoms = self.args.num_atoms

        if not self.args.no_noisy:
            fc_layer = NoisyLinear
        else:
            fc_layer = nn.Linear

        self.output_multiplier = self.num_atoms if not self.args.no_distr else 1
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        #no dueling
        self.fc1 = fc_layer(64 * 7 * 7, 512)
        self.fc2 = fc_layer(512, num_actions * self.output_multiplier)

        #dueling
        self.fc_value = nn.Sequential(
            fc_layer(64 * 7 * 7, 512),
            nn.ReLU(),
            fc_layer(512, 1 * self.output_multiplier),
        )
        self.fc_advantage = nn.Sequential(
            fc_layer(64 * 7 * 7, 512),
            nn.ReLU(),
            fc_layer(512, num_actions * self.output_multiplier),
        )
            
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)

        if self.args.no_dueling:
            if not self.args.no_distr:  # Distributional output
                x = x.view(-1, self.num_actions, self.args.num_atoms)
                x = torch.softmax(x, dim=2)  # Ensure probabilities over atoms
            x = self.fc1(x)
            x = self.fc2(x)
            return x
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        if not self.args.no_distr:
            value = value.view(-1, 1, self.num_atoms)
            advantage = advantage.view(-1, self.num_actions, self.num_atoms)
            
            x = value + (advantage - advantage.mean(dim=1, keepdim=True))
            x = torch.softmax(x, dim=2)  # Ensure probabilities over atoms
            # print("Q Net output: ", x)
            # print("Q Net output shape: ", x.shape)        
        else:
            # Standard dueling 
            x = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return x