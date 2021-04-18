from collections import namedtuple
import random
import numpy as np
from torch.utils.data import Dataset

Transition = namedtuple(
    'Transition', ('state', 'action', 'log_prob', 'entropy', 'reward', 'vs_t', 'next_state'))


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=Transition)
        self.capacity = capacity
        # Used for push
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

