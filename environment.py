import gym
from wrappers.frame_skipper import FrameSkipper
from gym.wrappers import FrameStack, GrayScaleObservation


class CarRacingEnv:

    def __init__(self, device):
        super().__init__()
        self.total_rew = 0
        self.state = None
        self.done = False
        self.device = device

        self.env = gym.make("CarRacing-v0")
        self.env = GrayScaleObservation(self.env)
        self.env = FrameStack(self.env, 4)
        self.env = FrameSkipper(self.env, 4)
        print(self.env.observation_space)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.state, self.done, self.total_rew = self.env.reset(), False, 0

    def rand_episode_run(self):
        while not self.done:
            self.env.render()
            ac = self.select_action(self.state)
            self.state, rew, self.done, info = self.env.step(ac)
            print(self.state)
            print(self.state.shape)
            self.total_rew += rew

    def print_reward(self):
        print('Total Reward obtained: {0}'.format(self.total_rew))

    def spec(self):
        return self.env.spec

    def close(self):
        self.close()


