import cnn 
import copy
import math

from cnn import CNN
import random
import torch

EPS_DECAY = 0.995
EPS_MIN = 0.001
ETA_MIN = 0.0000001
ETA_DECAY = 0.175

class Agent:
    def __init__(self, act_space, obs_space, eta):
        self.eta = eta
        self.act_space = act_space
        self.obs_space = obs_space
        self.cnn = CNN(obs_space.shape[0], act_space.n)
        self.optimizer = torch.optim.Adam(self.cnn.model.parameters(), lr=self.eta, weight_decay=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1.0
        self.steps = 0
        self.gamma = 0.999
        self.target_net = copy.deepcopy(self.cnn)
        self.loss = None
        # evaluate the network
        #self.cnn.model.eval()
        self.target_net.model.eval()


    def act(self, observation):
        # define epsilon
        self.epsilon = max(self.epsilon * EPS_DECAY, EPS_MIN)
        # increment steps
        self.steps += 1
        # create observation tensor
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)

        # with no grad
        with torch.no_grad():
            # get q values for the observation
            q_values = self.cnn(obs)

        # greedy policy
        if random.random() < self.epsilon:
            # random action
            return self.act_space.sample()
        else:
            # best action
            return q_values.argmax().item()

    def learn(self, batch):
        # train the network
        self.cnn.model.train()
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
        for param in self.cnn.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.soft_update(0.01)
        #if self.steps % 40 == 0:
        #self.hard_update()

    def soft_update(self, tau):
        for target_param, param in zip(self.target_net.parameters(), self.cnn.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update(self):
        self.target_net.load_state_dict(self.cnn.state_dict())