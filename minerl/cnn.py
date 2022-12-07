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
        randomPick = random.sample(self.buffer, size)
        # on retourne les tenseurs
        return CNNInt(*zip(*randomPick))

# -------------------- CNN -------------------------------------
class CNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__() 
        input_width, input_height, input_channels = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten()
        )
        '''self.layer3 = nn.Sequential(
            nn.Conv2d(5, 1, (2, 2), stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )'''
        self.layer4 = nn.Sequential(
            nn.Linear(7*7*32, 128),
            nn.ReLU())
            
        self.layer5 = nn.Sequential(
            nn.Linear(128, output_size),
            nn.ReLU())

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    #Utilisation du DQN
    def forward(self, x):
        x = torch.tensor(x)
        #print(x.shape) 


        if(x.dim()==3):
            x = x[None, :]
        x = torch.swapaxes(x, 1, 3)
        x = torch.swapaxes(x, 2, 3)
        x1 = self.layer1(x.float())
        x2 = self.layer2(x1)
        #x3 = self.layer3(x2) 
        x4 = self.layer4(x2)
        x5 = self.layer5(x4)

        return x5