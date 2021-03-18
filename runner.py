import torch
import numpy as np

from policy import Policy
from actions import get_action

class Runner:
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.input_channels = config['stack_frames']
        self.device = config['device']
        self.action_set = get_action(config['action_set_num'])
        self.policy = Policy(self.input_channels, len(self.action_set))
        self.last_epoch, optim_params, self.running_reward = self.policy.load_checkpoint(config['params_path'])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])
        if optim_params is not None:
            self.optimizer.load_state_dict(optim_params)

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
        return self.action_set[action.item()]

    def run(self):
        state, done, total_rew = self.env.reset(), False, 0
        while not done:
            self.env.render()
            action = self.select_action(state)
            state, rew, done, info = self.env.step(action)
            total_rew += rew
        print('Cumulative reward:', total_rew)
        self.env.close()
