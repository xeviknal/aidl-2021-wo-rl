import torch
import torch.nn as nn
from os import path


class ActorCriticPolicy(nn.Module):

    def __init__(self, inputs=4, actor_output=8, critic_output=1):
        super(ActorCriticPolicy, self).__init__()
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
        )

        # actor's layer
        self.actor_head = nn.Linear(128, actor_output)

        # critic's layer
        self.critic_head = nn.Linear(128, critic_output)

        self.activation = nn.Softmax(dim=-1)

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def forward(self, x):
       
        x = self.pipeline(x)
        # actor: chooses action to take from state s_t
        # by returning probability of each action
        action_prob = self.activation(self.actor_head(x))

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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            params_descriptor = torch.load(params_path, map_location=torch.device(device))
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
