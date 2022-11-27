import torch
import collections
import random
# ------------------- MEMORY REPLAY -------------------------------
DQNInt = collections.namedtuple('DQNInt', ('state', 'action', 'nextState', 'reward', 'end'))


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
        self.buffer.append(DQNInt(state, action, nextState, reward, end))

    def randomPick(self, size):
        # on tir aleatoirement des interactions distinctes
        return random.sample(self.buffer, size)

# -------------------- DQN -------------------------------------
class DQN(torch.nn.Module):
    def __init__(self, num_frames, num_outputs):
        super(DQN, self).__init__()
        # Network artchitecture
        self.network = nn.Sequential(
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
    def forward(self, x):
        # forward propagation
        return self.network(x)