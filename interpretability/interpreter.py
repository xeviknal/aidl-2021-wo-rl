import torch.nn as nn
from time import time


class Interpreter(nn.Module):
    """
    Include this module into your network to inspect the convolutional layer.
    It sends to the writer (only tested with Tensorboard) one image of the self.shape dimension.
    To avoid overhead, the images are pushed every 60 seconds (should be parameterized or using different reference than time).
    The implementation is far from ideal, but helps to understand what is going on through the convolutional.
    Usage:
    self.pipeline = nn.Sequential(
            Interpreter(writer, 'before_conv', [4, 96, 96]),
            nn.Conv2d(inputs, 12, kernel_size=3, stride=2, padding=1),  # [12, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [12, 24, 24]
            Interpreter(writer, '1st_conv', [12, 48, 48]),
            nn.Conv2d(12, 24, kernel_size=3),  # [24, 22, 22]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [24, 11, 11]
            Interpreter(writer, '2nd_conv', [24, 22, 22]),
        )
    """

    def __init__(self, writer, name, shape):
        super(Interpreter, self).__init__()
        self.writer = writer
        self.name = name
        self.shape = shape

    def forward(self, state):
        timestamp = int(time())
        if self.writer and timestamp % 60 == 0:
            for i, step in enumerate(state.reshape(self.shape)):
                self.writer.add_image('state/{}'.format(self.name),
                                      step.reshape(1, self.shape[1], self.shape[2]),
                                      timestamp + i)
        return state
