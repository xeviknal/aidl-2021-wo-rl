from helpers import wrap_env
import gym


class CarRacingEnv:

    def __init__(self):
        super().__init__()
        self.env = wrap_env(gym.make("CarRacing-v0"))
        self.total_rew = 0

    def reset(self):
        self.env.reset()

    def episode_run(self):
        self.env.render()
        ac = self.env.action_space.sample()
        ob, rew, done, info = self.env.step(ac)
        self.total_rew += rew

    def close(self):
        self.close()


