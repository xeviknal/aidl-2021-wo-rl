import torch
import torch.nn as nn
from os import path


class Policy(nn.Module):

    def __init__(self, inputs=4, outputs=8):
        super(Policy, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(inputs, 32, 3),  # [32, 94, 94]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 47, 47]
            nn.Conv2d(32, 64, 4),  # [64, 44, 44]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 22, 22]
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, outputs),
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
                #print('debug - params was found in params-descriptor')
                #print(f"params_descriptor['params'] = {params_descriptor['params']}")
                self.load_state_dict(params_descriptor['params'])
                epoch = params_descriptor['epoch']
                #print(f"epoch = {epoch}")
            else:
                #print('debug - params was NOT found in params-descriptor')
                self.load_state_dict(params_descriptor)

            print("Model params are loaded now")
        else:
            print("Params not found: training from scratch")

        return epoch

    def save_checkpoint(self, params_path, epoch):
        #print(f'Saving checkpoint in epoch {epoch}')
        torch.save({
            'epoch': epoch,
            'params': self.state_dict(),
        }, params_path)
        print("Relax, params are saved now")
