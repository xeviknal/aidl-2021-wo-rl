import torch
from environment import CarRacingEnv
from helpers import show_video, display_start

# if gpu is to be used
device = torch.device("cuda") if False else torch.device("cpu")

env = CarRacingEnv(device)
env.reset()
env.rand_episode_run()
#env.close()
env.print_reward()
show_video()

if __name__ == "__main__":
    config = { }
