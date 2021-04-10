import gym
from wrappers.frame_skipper import FrameSkipper
from wrappers.early_stop import EarlyStop
from wrappers.green_penalty import GreenPenalty
from gym.wrappers import FrameStack, GrayScaleObservation, Monitor


class CarRacingEnv:

    def __init__(self, device, seed, stack_frames=4, train=False):
        super().__init__()
        self.total_rew = 0
        self.state = None
        self.done = False
        self.device = device
        self.train = train

        self.env = gym.make("CarRacing-v0")
        self.env.seed(seed)
        if not train:
            self.env = Monitor(self.env, './video', force=True)
        self.env = GreenPenalty(self.env)
        self.env = GrayScaleObservation(self.env)
        self.env = FrameStack(self.env, stack_frames)
        self.env = FrameSkipper(self.env, 4)
        self.env = EarlyStop(self.env, 50)

        print(self.env.observation_space)

    def max_episode_steps(self):
        return self.spec().max_episode_steps

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.state, self.done, self.total_rew = self.env.reset(), False, 0

    def print_reward(self):
        print('Total Reward obtained: {0}'.format(self.total_rew))

    def spec(self):
        return self.env.spec

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()


