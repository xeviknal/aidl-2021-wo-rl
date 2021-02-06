import torch

from environment import CarRacingEnv
from trainer import Trainer

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# if gpu is to be used
device = torch.device("cuda") if False else torch.device("cpu")

if __name__ == "__main__":
    hyperparams = {
        'num_episodes': 20000,  # Number of training episodes
        'lr': 1e-2,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'params_path': './params/policy-params.dl'
    }

    env = CarRacingEnv(device, hyperparams['stack_frames'])
    trainer = Trainer(env, hyperparams)
    trainer.train()

