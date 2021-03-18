import torch
import numpy as np

from policy import Policy
from actions import available_actions

import seaborn as sns
import matplotlib.pyplot as plt
import os

class Runner:
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.input_channels = config['stack_frames']
        #self.device = config['device']
        self.policy = Policy(self.input_channels, len(available_actions))
        self.policy.load_checkpoint(config['params_path'])

    def select_action(self, state):
        if state is None:  # First state is always None
            # Adding the starting signal as a 0's tensor
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        # We pick the action from a sample of the probabilities
        # It prevents the model from picking always the same action
        m = torch.distributions.Categorical(probs)
        eprobs = torch.exp(torch.squeeze(probs)).detach().numpy()
        action = m.sample()
        return available_actions[action.item()], eprobs

    def run(self):
        state, done, total_rew = self.env.reset(), False, 0
        eprobs_list = []
        i = 0
        while not done and i<50:
            self.env.render()
            action, eprobs = self.select_action(state)
            state, rew, done, info = self.env.step(action)
            total_rew += rew
            eprobs_list.append(eprobs)
            eprobs_list_T = list(map(list, zip(*eprobs_list)))
            plt.figure()
            heatmap = sns.heatmap(eprobs_list_T, vmin=0, vmax=1, cmap="Blues")
            plt.savefig("./heatmap/heatmap_{}".format(i))
            i = i +1
        os.system("ffmpeg -r 20 -i ./heatmap/heatmap_%01d.png -vcodec mpeg4 -y ./heatmap/heatmap.mp4")
        print('Cumulative reward:', total_rew)
        self.env.close()