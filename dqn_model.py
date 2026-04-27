import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for Deep Q-Learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        Input: state tensor
        Output: Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class CNNDQN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for Deep Q-Learning from image inputs.
    """
    def __init__(self, action_dim, n_channels=1):
        super(CNNDQN, self).__init__()
        # Input shape: (n_channels, 64, 64) - Grayscale 64x64 stack
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=5, stride=2) # -> 16, 30, 30
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)          # -> 32, 13, 13
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)          # -> 32, 5, 5

        # 32 * 5 * 5 = 800 -> hidden layer -> action_dim
        self.fc1 = nn.Linear(800, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        """
        Forward pass.
        x: tensor of shape (batch, n_channels, 64, 64)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)