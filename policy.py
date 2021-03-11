import torch
import torch.nn as nn
from os import path


class Policy(nn.Module):

    def __init__(self, inputs=4, outputs=8):
        super(Policy, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(inputs, 12, kernel_size=3, stride=2, padding=1),  # [12, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [12, 24, 24]
            nn.Conv2d(12, 24, kernel_size=3),  # [24, 22, 22]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [24, 11, 11]
            nn.Conv2d(24, 32, 4),  # [32, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 4, 4]
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),  # [ 512, 256 ]
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outputs),
            nn.LogSoftmax(dim=-1)
        )
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.pipeline(x)

    def load_checkpoint(self, params_path):
        epoch = 0
        if path.exists(params_path):
            params_descriptor = torch.load(params_path)
            epoch = 0
            if 'params' in params_descriptor:
                self.load_state_dict(params_descriptor['params'])
                epoch = params_descriptor['epoch']
            else:
                self.load_state_dict(params_descriptor)

            print("Model params are loaded now")
        else:
            print("Params not found: training from scratch")

        return epoch

    def save_checkpoint(self, params_path, epoch):
        torch.save({
            'epoch': epoch,
            'params': self.state_dict(),
        }, params_path)
        print("Relax, params are saved now")
