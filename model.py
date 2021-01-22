import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, inputs, outputs, hidden_layer=128):
        super(DQN, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(inputs, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, outputs),
        )

    def forward(self, x):
        return self.pipeline(x)
