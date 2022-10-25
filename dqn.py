import torch

class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size, env):
        self.model = torch.nn.Sequential( torch.nn.linear(input_size, 128),
                                          torch.nn.ReLU(), 
                                          torch.nn.linear(128, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.linear(128, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, output_size) )
        self.loss = torch.nn.MSELoss() 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.env = env
        
    def train(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
    
    def predict(self, x, y):
        observation = self.env.reset()
        for _ in range(1000):
            observation, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            # reward_cumul += reward

            if terminated or truncated:
                pass
              #  nb_episodes += 1
              #  x.append(nb_episodes)
               # y.append(reward_cumul)
                #reward_cumul = 0
               # observation, info = observation.reset()