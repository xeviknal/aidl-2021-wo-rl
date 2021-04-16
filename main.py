import torch
import numpy as np
import helpers
from environment import CarRacingEnv
from trainer import Trainer
from runner import Runner

from pyvirtualdisplay import Display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hyperparams = {
        'num_episodes': 150000,  # Number of training episodes
        'lr': 1e-5,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'params_path': './params/policy-params.dl',
        'train': True
    }

    # Reproducibility: manual seeding
    seed = 7081960
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


    env = CarRacingEnv(device, hyperparams['stack_frames'], hyperparams['train'])
    helpers.display_start()
    if hyperparams['train']:
        trainer = Trainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
        

