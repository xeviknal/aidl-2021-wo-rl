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
        action = m.sample()
        return available_actions[action.item()], probs

    def run(self):
        state, done, total_rew = self.env.reset(), False, 0
        probs_list = []
        i = 0
        while not done:
            self.env.render()
            action, probs = self.select_action(state)
            state, rew, done, info = self.env.step(action)
            total_rew += rew
            probs_list.append((torch.squeeze(probs)).detach().numpy())
            probs_list_T = list(map(list, zip(*probs_list)))
#            plt.figure(figsize=(400,600))
            plt.figure()
            heatmap = sns.heatmap(probs_list_T, vmin=0, vmax=1, cmap="Blues")
            plt.savefig("./heatmap/heatmap_{}".format(i))
            plt.close()
            i = i +1            
        os.system("ffmpeg -r 50 -i ./heatmap/heatmap_%01d.png -vcodec mpeg4 -y ./heatmap/heatmap.mp4")        
        os.system("ffmpeg -i ./heatmap/heatmap.mp4 -vf scale=600:400 -strict -2 ./heatmap/heatmap_scale.mp4")
#        os.system("ffmpeg -i ./video/*.mp4 -vcodec copy -acodec copy -movflags faststart ./video/car_no_scale.mp4")
#        os.system("ffmpeg -i ./video/car_no_scale.mp4 -vf scale=640:480 -strict -2 ./video/car_scale.mp4")
        os.system("ffmpeg -i ./video/*.mp4 -i ./heatmap/heatmap_scale.mp4 -filter_complex hstack ./video/car_and_heatmap.mp4")
        print('Cumulative reward:', total_rew)
        self.env.close()