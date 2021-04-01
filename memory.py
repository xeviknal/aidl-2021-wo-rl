from collections import namedtuple
import random
import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'log_prob', 'entropy', 'reward', 'vs_t', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.memory = np.empty(capacity, dtype=Transition)
        self.capacity = capacity
        self.batch_size = batch_size
        self.current_batch_starts = 0
        # Used for push
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def get_batch(self):
        # Compute a safe end of the batch: the last batch might be shorter
        end = self.current_batch_starts + self.batch_size - 1
        if end >= self.capacity:
            end = self.capacity - 1

        # Computing the transitions for the batch
        transitions = self.memory[self.current_batch_starts:end]
        self.current_batch_starts += self.batch_size
        # Round reading: after the last batch, we start over
        if self.current_batch_starts >= self.capacity:
            self.current_batch_starts = 0

        return transitions

    def shuffle(self):
        np_mem = np.array(self.memory)
        np.random.shuffle(np_mem)
        self.memory = np_mem  # .tolist()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.position = 0
        self.current_batch_starts = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)
