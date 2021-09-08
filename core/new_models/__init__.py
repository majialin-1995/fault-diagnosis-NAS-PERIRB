import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os

# Target Model definition
models_path = './models/'


class target_net(nn.Module):
    def __init__(self):
        super(target_net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(127488, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Dropout(0.5)(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x