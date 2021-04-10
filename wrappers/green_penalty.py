from gym import Wrapper
import numpy as np


class GreenPenalty(Wrapper):
    r"""Stops the episode after n steps with negative reward"""

    def __init__(self, env):
        super(GreenPenalty, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if np.mean(state[:, :, 1]) > 180.0:
            reward -= 0.05
        return state, reward, done, info
