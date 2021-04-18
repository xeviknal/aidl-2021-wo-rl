import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import helpers
from actions import get_action
from trainers.factory import get_strategy_descriptor


def build(strategy, env, hyper_params):
    _, policy_class = get_strategy_descriptor(strategy)
    return Runner(env, hyper_params, policy_class)

class Runner:
    def __init__(self, env, config, policy_class):
        super().__init__()
        self.env = env
        self.config = config
        self.build_heatmap = config['heatmap']
        self.input_channels = config['stack_frames']
        self.action_set = get_action(config['action_set_num'])
        self.policy = policy_class(self.input_channels, len(self.action_set))
        self.policy.load_checkpoint(config['params_path'])
        self.heatmap_probs = []

    def select_action(self, state):
        if state is None:  # First state is always None
            # Adding the starting signal as a 0's tensor
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)[0]
        # We pick the action from a sample of the probabilities
        # It prevents the model from picking always the same action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return self.action_set[action.item()], probs

    def heatmap_build(self, iteration, step_probs):
        self.heatmap_probs.append((torch.squeeze(step_probs)).detach().numpy())
        probs_list_t = list(map(list, zip(*self.heatmap_probs)))
        plt.figure()
        sns.heatmap(probs_list_t, vmin=0, vmax=1, cmap="Blues")
        plt.savefig("./heatmap/heatmap_{}".format(iteration))
        plt.close()

    def compose_heatmap(self):
        os.system("ffmpeg -y -r 50 -i ./heatmap/heatmap_%01d.png -vcodec mpeg4 ./heatmap/heatmap.mp4")
        os.system("ffmpeg -y -i ./heatmap/heatmap.mp4 -vf scale=600:400 -strict -2 ./heatmap/heatmap_scale.mp4")
        os.system("ffmpeg -y -i ./video/openaigym.*.mp4 -i ./heatmap/heatmap_scale.mp4 -filter_complex hstack ./video/car_and_heatmap.mp4")
        print('####################################################################################')
        print('In case the previous `ffmpeg` commands didnt work, please execute the following:')
        print("ffmpeg -y -i ./video/openaigym.*.mp4 -i ./heatmap/heatmap_scale.mp4 -filter_complex hstack ./video/car_and_heatmap.mp4")
        print('####################################################################################')

    def run(self):
        if self.build_heatmap:
            helpers.create_directory('heatmap')

        state, done, total_rew, i = self.env.reset(), False, 0, 0
        while not done:
            self.env.render()
            action, probs = self.select_action(state)
            state, rew, done, info = self.env.step(action)
            total_rew += rew
            if self.build_heatmap:
                self.heatmap_build(i, probs)
            i += 1
        self.env.close()

        if self.build_heatmap:
            self.compose_heatmap()

        print('Cumulative reward:', total_rew)
