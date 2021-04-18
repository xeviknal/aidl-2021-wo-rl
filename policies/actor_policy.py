import torch
import torch.nn as nn
from os import path


class ActorPolicy(nn.Module):

    def __init__(self, inputs=4, outputs=8):
        super(ActorPolicy, self).__init__()
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
            nn.Softmax(dim=-1)
        )
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def forward(self, x):
        return self.pipeline(x)

    def load_checkpoint(self, params_path):
        epoch = 0
        running_reward = 10
        optim_params = None
        if path.exists(params_path):
            params_descriptor = torch.load(params_path)
            epoch = 0
            running_reward = 0
            if 'params' in params_descriptor:
                self.load_state_dict(params_descriptor['params'])
                optim_params = params_descriptor['optimizer_params']
                epoch = params_descriptor['epoch']
                running_reward = params_descriptor['running_reward']
            else:
                self.load_state_dict(params_descriptor)

            print("Model params are loaded now")
        else:
            print("Params not found: training from scratch")

        return epoch, optim_params, running_reward

    def save_checkpoint(self, params_path, epoch, running_reward, optimizer):
        torch.save({
            'epoch': epoch,
            'params': self.state_dict(),
            'running_reward': running_reward,
            'optimizer_params': optimizer.state_dict(),
        }, params_path)
        print("Relax, params are saved now")
