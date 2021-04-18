import time
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
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    config['params_path'] = f'./params/policy-params-{experiment}-{int(time.time())}.dl'

    # make sure that params folder exists
    helpers.create_directory('params')

    env = CarRacingEnv(device, seed, config['stack_frames'], config['train'])
    helpers.display_start()
    # Train it first
    trainer = Trainer(env, config)
    trainer.train()

    # Let's store a vid with one episode
    config['train'] = False
    runner = Runner(env, config)
    runner.run()
    config['train'] = True


# for concurrent runs and logging
experiment = 'ppo-nm-hp-tuning-max'
if __name__ == "__main__":
    hyperparams = {
        'num_epochs': 1500,  # Number of training episodes
        'num_ppo_epochs': tune.randint(3, 5),
        'mini_batch_size': 128,
        'memory_size': 2000,
        'eps': 0.2,
        'c1': tune.quniform(1.5, 2.5, 0.5),  # Value Function coeff
        'c2': tune.quniform(0.00, 0.06, 0.02),  # Entropy coeff
        'lr': 1e-3,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 10,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'experiment': experiment,
        'action_set_num': 4,
        'train': True,
        'seed': 190421
    }

analysis = tune.run(
    train,
    metric='running_reward',
    mode='max',
    num_samples=3,
    resources_per_trial={"cpu": 0.4, "gpu": 0.3},
    config=hyperparams,
)

print("Best config: ", analysis.get_best_config(
    metric="running_reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
