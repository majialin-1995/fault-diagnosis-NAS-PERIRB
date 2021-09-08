import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os

class Conv1dModel(nn.Module):
    def __init__(self):
        super(Conv1dModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(64*392, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Dropout(0.25)(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x