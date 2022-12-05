import cnn 
import copy
import math
import numpy as np
from cnn import CNN
import random
import torch
from skimage.color import rgb2gray

EPS_DECAY = 0.995
EPS_MIN = 0.001
ETA_MIN = 0.0000001
ETA_DECAY = 0.175

class Agent:
    def __init__(self, act_space, obs_space, eta):
        self.eta = eta
        self.act_space = act_space
        self.obs_space = obs_space
        self.im_height = 84
        self.im_width = 84
        self.cnn = CNN(1, self.im_height, self.im_width, len(act_space))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.eta, weight_decay=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1.0
        self.steps = 0
        self.gamma = 0.999
        self.target_net = copy.deepcopy(self.cnn)
        self.loss = None
        
        # evaluate the network
        # self.cnn.model.eval()
        self.target_net.eval()


    def rgb2gray(self, obs):
        state = (
            resize(rgb2gray(state), (self.im_height,
                                    self.im_width), mode="reflect")
            * 255
        )
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=device, dtype=torch.float)


    def act(self, observation):
        # define epsilon
        self.epsilon = max(self.epsilon * EPS_DECAY, EPS_MIN)
        # increment steps
        self.steps += 1
        # create observation tensor
        # obs = torch.tensor(observation.copy(), dtype=torch.float32, device=self.device)
        #obs = torch.from_numpy(observation.copy()).unsqueeze(0).to(self.device)
        # with no grad
        with torch.no_grad():
            # get q values for the observation
            q_values = self.cnn(observation)
           
        # greedy policy
        if random.random() < self.epsilon:
            # random action
            return self.act_space.sample()
        else:
            with torch.no_grad():
                # best action
                #print(self.cnn(observation))
                return torch.argmax(q_values.item())

    def learn(self, batch):
        # train the network
        self.cnn.train()
        # create tensors for the batch
        states = torch.tensor([b.state for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([b.nextState for b in batch], dtype=torch.float32, device=self.device)
        ends = torch.tensor([b.end for b in batch], dtype=torch.float32, device=self.device)

        # loss function
        loss_fn = torch.nn.MSELoss()

        # get q values for the states
        q_values = self.cnn(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # get q values for the next states
        next_q_values = self.target_net(next_states)
        next_q_values = next_q_values.detach().max(1)[0]

        # Bellman's equation
        targets = rewards + self.gamma * next_q_values * (1 - ends)

        # compute the loss
        loss = loss_fn(q_values, targets)
        self.loss = loss.item()
        # optimize the network and update weights
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.cnn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
       # self.soft_update(0.01)
        #if self.steps % 40 == 0:
        #self.hard_update()

    def soft_update(self, tau):
        for target_param, param in zip(self.target_net.parameters(), self.cnn.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update(self):
        self.target_net.load_state_dict(self.cnn.state_dict())