from collections import namedtuple
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'log_prob', 'advantage', 'vs_t', 'entropy'))


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        self.current_batch_starts = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def get_batch(self):
        transitions = self.memory[self.current_batch_starts:self.current_batch_starts + self.batch_size]
        self.current_batch_starts += self.batch_size
        return transitions

    def from_indexes(self, index_ids):
        return self.memory[index_ids]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.position = 0
        self.current_batch_starts = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)
