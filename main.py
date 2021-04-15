import torch
import numpy as np

import helpers
from environment import CarRacingEnv
from runner import Runner
from trainers.ppo_trainer import PPOTrainer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = 'dev'
if __name__ == "__main__":
    hyperparams = {
        # Run management params
        'experiment': experiment,
        'params_path': f'./params/policy-params-{experiment}.dl',
        'runs_path': f'./runs/{experiment}',
        'log_interval': 10,  # controls how often we log progress
        'device': device,
        'train': True,
        # Train management
        'num_epochs': 25000,  # Number of training episodes
        'num_ppo_epochs': 10,
        'mini_batch_size': 128,
        'memory_size': 2000,
        'eps': 0.2,
        'c1': 1.,  # Value Function coeff
        'c2': 0.01,  # Entropy coeff
        'lr': 1e-3,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'stack_frames': 4,
        'action_set_num': 0,
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
        trainer = PPOTrainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
