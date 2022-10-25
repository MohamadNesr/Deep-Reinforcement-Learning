import collections
import random

# Nos tuples d'interactions
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