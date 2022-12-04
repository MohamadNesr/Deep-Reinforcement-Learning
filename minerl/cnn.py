import torch
import torch.nn as nn
import collections
import random

# ------------------- MEMORY REPLAY -------------------------------
CNNInt = collections.namedtuple('CNNInt', ('state', 'action', 'nextState', 'reward', 'end'))


class ExperienceReplay:
    def __init__(self):
        self.size = 100000
        self.buffer = []

    def save(self, state, action, nextState, reward, end):
        # si la taille du buffer est superieure a la taille maximale
        if len(self.buffer) > self.size:
            # on enleve la premiere interaction
            self.buffer.pop(0)
        # on ajoute la nouvelle interaction
        self.buffer.append(CNNInt(state, action, nextState, reward, end))

    def randomPick(self, size):
        # on tir aleatoirement des interactions distinctes
        return random.sample(self.buffer, size)

# -------------------- CNN -------------------------------------
class CNN(torch.nn.Module):
    def __init__(self, num_frames, num_outputs):
        super(CNN, self).__init__()
        # Network artchitecture
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64,128),
            torch.nn.ReLU(),
            nn.Linear(128,num_outputs)
        )
        self.loss = torch.nn.MSELoss()
        self.param = torch.nn.ModuleList(self.model.children())
    
    def forward(self, x):
        # forward propagation
        f = self.model(x)
        return f
