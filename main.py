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
        'num_epochs': 25000,  # Number of training episodes
        'num_ppo_epochs': 10,
        'mini_batch_size': 128,
        'memory_size': 2000,
        'eps': 0.2,
        'c1': 0.3,  # TODO: To be defined
        'c2': 0.4,  # TODO: To be defined
        'lr': 1e-2,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 100,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'params_path': './params/policy-params.dl',
        'train': True
    }

    # make sure that params folder exists
    helpers.create_directory('params')

    env = CarRacingEnv(device, hyperparams['stack_frames'], hyperparams['train'])
    helpers.display_start()
    if hyperparams['train']:
        trainer = Trainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
