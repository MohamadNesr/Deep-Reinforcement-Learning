import collections
import numpy as np
import random

DQNint = collections.namedtuple('DQNint', ['state','action','nextState', 'reward','end'])

class ExperienceReplay:
    def __init__(self):
        self.size = 100000
        self.buffer = []

    def save(self, state, action, nextState, reward, end):
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        self.buffer.append(DQNint(state, action, nextState, reward, end))

    def randomPick(self, size):
        return random.sample(self.buffer, size)
