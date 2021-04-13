import torch
import numpy as np
from ray import tune

import helpers
from environment import CarRacingEnv
from runner import Runner
from trainer import Trainer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config):
    # Reproducibility: manual seeding
    seed = 7081960  # Yann LeCun birthday
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # make sure that params folder exists
    helpers.create_directory('params')

    env = CarRacingEnv(device, seed, config['stack_frames'], config['train'])
    helpers.display_start()
    if config['train']:
        trainer = Trainer(env, config)
        trainer.train()
    else:
        runner = Runner(env, config)
        runner.run()


# for concurrent runs and logging
experiment = 'ppo-nm-hp-tuning'
if __name__ == "__main__":
    hyperparams = {
        'num_epochs': 1000,  # Number of training episodes
        'num_ppo_epochs': 6,
        'mini_batch_size': 128,
        'memory_size': 2000,
        'eps': 0.2,
        'c1': tune.quniform(0.5, 2.5, 0.25),  # Value Function coeff
        'c2': tune.quniform(0.01, 0.15, 0.01),  # Entropy coeff
        'lr': 1e-3,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'experiment': experiment,
        'params_path': f'./params/policy-params-{experiment}.dl',
        'action_set_num': 0,
        'train': True,
    }

analysis = tune.run(
    train,
    metric='running_reward',
    mode='min',
    num_samples=10,
    resources_per_trial={"gpu": 1},
    config=hyperparams,
)
