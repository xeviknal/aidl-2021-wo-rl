from helpers import wrap_env
import gym
import random
import torch
import math


def compute_eps_threshold(step, eps_start, eps_end, eps_decay):
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)


class CarRacingEnv:

    def __init__(self, device):
        super().__init__()
        self.env = wrap_env(gym.make("CarRacing-v0"))
        self.total_rew = 0
        self.device = device

    def reset(self):
        self.env.reset()

    def rand_episode_run(self):
        self.env.render()
        ac = self.env.action_space.sample()
        ob, rew, done, info = self.env.step(ac)
        self.total_rew += rew

    def print_reward(self):
        print('Total Reward obtained: {0}'.format(self.total_rew))

    def close(self):
        self.close()

    def select_action(self, policy, state, eps_greedy_threshold, n_actions):
        # e-greedy strategy
        if random.random() > eps_greedy_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = policy(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(n_actions)]], device=self.device, dtype=torch.long)
        return action

