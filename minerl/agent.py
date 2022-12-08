import cnn 
import copy
import math
import numpy as np
from cnn import CNN
import random
import torch
from skimage.color import rgb2gray
from skimage.transform import resize

EPS_DECAY = 0.99995
EPS_MIN = 0.1
ETA_MIN = 0.0000001
ETA_DECAY = 0.175
action_keys = ["attack", "left", "right"]

class Agent:
    def __init__(self, act_space, obs_space, eta):
        self.eta = eta
        self.act_space = act_space
        self.obs_space = obs_space
        self.im_height = 224
        self.im_width = 224
        self.cnn = CNN(obs_space['pov'].shape , len(act_space))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.eta, weight_decay=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1.0
        self.steps = 0
        self.gamma = 0.999
        self.target_net = copy.deepcopy(self.cnn)
        self.loss = 0
        
        # evaluate the network
        # self.cnn.model.eval()
        self.target_net.eval()

    def process_state(self, state):
        # convert to grayscale
        state = rgb2gray(state)
        # resize
        state = resize(state, (self.im_height, self.im_width), anti_aliasing=True)
        # normalize
        state = state.astype(np.float32)
        state /= 255.0
        return state

    def act(self, state):
        self.cnn.eval()
        # define epsilon
        self.epsilon = max(self.epsilon * EPS_DECAY, EPS_MIN)
        # increment steps
        self.steps += 1
        # state = self.process_state(state)
        #print(state.shape)
        #input = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self.device)
        #print(input.shape)
        # get q values for the state
        #with torch.no_grad():
        a = "none"

        # act_array = [act[action_key] for action_key in action_keys]
        # get the action
        action = self.act_space.noop()
        # action['camera'] = 0,0
        if random.random() < self.epsilon:
            # choose random action
            action = dict(action)
            a = "random"
            action['attack'] = random.randint(0, 1)
            if (action['attack'] == 1):
                action['left'] = 0
                action['right'] = 0
            else:
                action['left'] = random.randint(0, 1)
                if (action['left'] == 1):
                    action['right'] = 0
                else:
                    action['right'] = 1

            return action, a
        else:
            q_values = self.cnn(state)
            q_values = torch.tensor([q_values.argmax()])
            act = action_keys[q_values]    
            a = "best"     
            #print(act)   
            #action = act
            action = dict(action)
            action[act] = 1
            return action, a



    def learn(self, batch_size, exp_replay):
        # get batch
        batch = exp_replay.randomPick(batch_size)
        # prepare batch
        states = torch.stack([torch.tensor(state) for state in batch[0]])
        #print("states tensor built")
        #print(states.shape)
        #print(batch[1])
        #print("--------------------------------------------------------------------------------")
        tensors = [torch.tensor(list(v.values())) for v in batch[1]]
        actions = torch.stack(tensors)
        #print("actions tensor built")
        #print(actions.shape)
        #print(actions)
        #print("--------------------------------------------------------------------------------")
        next_states = torch.stack([torch.tensor(next_state) for next_state in batch[2]])
        #print("next_states tensor built")
        #print(next_states.shape)
        #print(next_states)
        #print("--------------------------------------------------------------------------------")
        rewards = torch.stack([torch.tensor(reward, dtype=None) for reward in batch[3]])
        #print("rewards tensor built")
        #print(rewards.shape)
        #print(rewards)
        #print("--------------------------------------------------------------------------------")
        dones = torch.stack([torch.tensor(done, dtype=torch.float32) for done in batch[4]])
        #print("dones tensor built")
        #print(dones.shape)
        #print(dones)
        #print("--------------------------------------------------------------------------------")
        
        target = self.cnn(states)
        target = target.gather(1, actions)
        target = torch.max(target, 1)[0]
        with torch.no_grad():
            next_target = self.target_net(next_states)
        
        self.cnn.train()
        targets = rewards + self.gamma * torch.max(next_target, 1)[0] * (1 - dones)
        loss = self.cnn.loss(target, targets)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.cnn.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target network
        self.soft_update(0.01)
        #print("learned")

    def soft_update(self, tau):
        for target_param, param in zip(self.target_net.parameters(), self.cnn.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update(self):
        self.target_net.load_state_dict(self.cnn.state_dict())


