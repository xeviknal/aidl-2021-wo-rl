import torch

import helpers
from environment import CarRacingEnv
from trainer import Trainer
from runner import Runner

from pyvirtualdisplay import Display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hyperparams = {
        'num_episodes': 40000,  # Number of training episodes
        'lr': 1e-2,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'params_path': './params/policy-params.dl',
        'eps': 0.4,
        'eps_decay_episodes': 250,
        'train': True
    }

    env = CarRacingEnv(device, hyperparams['stack_frames'], hyperparams['train'])
    helpers.display_start()
    if(hyperparams['train']):
        trainer = Trainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
        

