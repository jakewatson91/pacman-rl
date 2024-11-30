#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from dqn_model import NoisyLinear

class DistributionalModel(nn.Module):
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
        super(NoisyLinear, self).__init__()

        self.args = args

        #noisy
        if self.args.no_noisy:
            #standard
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, num_actions)

            #dueling
            self.fc_value = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, 1 * self.args.num_atoms),
            )
            self.fc_advantage = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions * self.args.num_atoms),
            )
        else: 
            #standard
            self.fc1 = NoisyLinear(64 * 7 * 7, 512)
            self.fc2 = NoisyLinear(512, num_actions)

            #dueling
            self.fc_value = nn.Sequential(
                NoisyLinear(64 * 7 * 7, 512),
                nn.ReLU(),
                NoisyLinear(512, 1 * self.args.num_atoms),
            )
            self.fc_advantage = nn.Sequential(
                NoisyLinear(64 * 7 * 7, 512),
                nn.ReLU(),
                NoisyLinear(512, num_actions * self.args.num_atoms),
            )
            