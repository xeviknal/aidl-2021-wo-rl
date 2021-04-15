import torch
import numpy as np

import helpers
from environment import CarRacingEnv
from runner import Runner
from trainers.reinforce_baseline_trainer import ReinforceBaselineTrainer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hyperparams = {
        'num_episodes': 200000,  # Number of training episodes
        'lr': 1e-3,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'params_path': './params/policy-params.dl',
        'action_set_num': 0,
        'train': True
    }

    # Reproducibility: manual seeding
    seed = 7081960  # Yann LeCun birthday
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # make sure that params folder exists
    helpers.create_directory('params')

    env = CarRacingEnv(device, seed, hyperparams['stack_frames'], hyperparams['train'])
    helpers.display_start()
    if hyperparams['train']:
        trainer = ReinforceBaselineTrainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()

