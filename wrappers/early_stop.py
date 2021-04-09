from gym import Wrapper
import numpy as np


class EarlyStop(Wrapper):
    r"""Stops the episode after n steps with negative reward"""
    def __init__(self, env, steps):
        super(EarlyStop, self).__init__(env)
        self.steps = steps
        self.remaining_steps = steps
        self.latest_rewards = []

    def reset(self, **kwargs):
        self.remaining_steps = self.steps
        self.latest_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.remaining_steps = self.remaining_steps - 1
        self.latest_rewards.append(reward)
        avg = 1
        if self.remaining_steps == 0:
            avg = np.array(self.latest_rewards).mean()
            if avg > 0:
                self.remaining_steps = self.steps
                self.latest_rewards = []
        return state, reward, avg < 0, info
