import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path


class Policy(nn.Module):

    def __init__(self, actor_output, critic_output, inputs=4):
        super(Policy, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(inputs, 16, 3),  # [32, 94, 94]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 47, 47]
            nn.Conv2d(16, 32, 4),  # [64, 44, 44]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 22, 22]
            nn.Conv2d(32, 64, 3),  # [64, 20, 20]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 10, 10]
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # actor's layer
        self.actor_head = nn.Linear(128, actor_output)

        # critic's layer
        self.critic_head = nn.Linear(128, critic_output)

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def forward(self, x):
       
        x= self.pipeline(x)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

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
