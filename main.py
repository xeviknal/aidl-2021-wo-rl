import torch
from environment import CarRacingEnv
from helpers import show_video

# if gpu is to be used
device = torch.device("cuda") if False else torch.device("cpu")

env = CarRacingEnv()
env.reset()
env.episode_run()
#env.close()
show_video()

if __name__ == "__main__":
    config = {}