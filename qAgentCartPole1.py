import math

from dqn import DQN
import random
import torch

class Agent:
    def __init__(self, act_space, obs_space, eta):
        self.eta = eta
        self.act_space = act_space
        self.obs_space = obs_space
        self.dqn = DQN(obs_space.shape[0], act_space.n)
        self.optimizer = torch.optim.SGD(self.dqn.model.parameters(), lr=eta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 0
        self.steps = 0
        self.gamma = 0.99

    def act(self, observation, EPS_START, EPS_END, EPS_DECAY):
        # define epsilon
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps / EPS_DECAY)
        # increment steps
        self.steps += 1
        # create observation tensor
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
        # evaluate the network
        self.dqn.model.eval()

        # with no grad
        with torch.no_grad():
            # get q values for the observation
            q_values = self.dqn(obs)

        # train the network
        self.dqn.model.train()

        # GREEDY method
        if random.random() < self.epsilon:
            # random action
            return self.act_space.sample()
        else:
            # best action
            return q_values.argmax().item()

    def learn(self, batch):
        # create tensors for the batch
        states = torch.tensor([b.state for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([b.nextState for b in batch], dtype=torch.float32, device=self.device)
        ends = torch.tensor([b.end for b in batch], dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.MSELoss()
        # evaluate the network
        self.dqn.model.eval()

        # get q values for the states
        q_values = self.dqn(states)

        # get q values for the actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # with no grad
        with torch.no_grad():
            # get q values for the next states
            next_q_values = self.dqn(next_states).detach().max(1)[0].unsqueeze(1)

        # train the network
        self.dqn.model.train()

        # Bellman's equation
        targets = rewards + self.gamma * next_q_values.max(1)[0] * (1 - ends)

        # compute the loss
        loss = loss_fn(q_values, targets)
        # optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update epsilon
        self.epsilon = max(0.01, self.epsilon - 1e-6)

