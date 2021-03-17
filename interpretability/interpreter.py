import torch.nn as nn
from time import time


class Interpreter(nn.Module):

    def __init__(self, writer, name, shape):
        super(Interpreter, self).__init__()
        self.writer = writer
        self.name = name
        self.shape = shape

    def forward(self, state):
        timestamp = int(time())
        if self.writer and timestamp % 5 == 0:
            for i, step in enumerate(state.reshape(self.shape)):
                self.writer.add_image('state/{}'.format(self.name),
                                      step.reshape(1, self.shape[1], self.shape[2]),
                                      timestamp + i)
        return state
