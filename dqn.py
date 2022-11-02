import torch

class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.model = torch.nn.Sequential( torch.nn.Linear(input_size, 32),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(32, 32),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(32, 32),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(32, output_size) )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.param = torch.nn.ModuleList(self.model.children())

    def forward(self, x):
        for f in self.param:
            x = f(x)
        return x
