import torch
import numpy as np

from policy import Policy
from actions import available_actions

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
        return available_actions[action.item()]

    def run(self):
        state, done, total_rew = self.env.reset(), False, 0
        while not done:
            self.env.render()
            action = self.select_action(state)
            state, rew, done, info = self.env.step(action)
            total_rew += rew
        print('Cumulative reward:', total_rew)
        self.env.close()