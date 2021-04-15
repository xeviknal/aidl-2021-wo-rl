import argparse
import torch
import numpy as np

import helpers
from environment import CarRacingEnv
from runner import Runner
import trainers.factory as trainer_factory

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="Name of the experiment", type=str, default="default")
parser.add_argument("--strategy", help="Name of the strategy to follow for training: vpg, baseline, ppo", type=str, default="vpg")
parser.add_argument("--log_interval", help="Checkpoint frequency", type=int, default=50)
parser.add_argument("--record", help="Runs the environment and records it", type=bool, default=False)
parser.add_argument("--epochs", help="Number of epochs to train", type=int, default=25000)
parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
parser.add_argument("--gamma", help="Discount factor", type=float, default=0.99)
parser.add_argument("--action_set", help="Action set: the set of actions that the policy will use", type=int, default=0)
parser.add_argument("--ppo_epochs", help="Number of proximal optimization epochs (Only for PPO training)", type=int, default=10)
parser.add_argument("--ppo_batch_size", help="Batch size for memory visit (Only for PPO training)", type=int, default=128)
parser.add_argument("--ppo_memory_size", help="Memory size (Only for PPO training)", type=int, default=2000)
parser.add_argument("--ppo_epsilon", help="Epsilon ratio (Only for PPO training)", type=float, default=0.2)
parser.add_argument("--ppo_value_coeff", help="Value Function Coeff (Only for PPO training)", type=float, default=1.)
parser.add_argument("--ppo_entropy_coeff", help="Entropy Coeff (Only for PPO training)", type=float, default=0.01)

args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hyperparams = {
        # Run management params
        'experiment': args.experiment,
        'strategy': args.strategy,
        'params_path': f'./params/policy-params-{args.experiment}.dl',
        'runs_path': f'./runs/{args.experiment}',
        'log_interval': args.log_interval,  # controls how often we log progress
        'device': device,
        'train': not args.record,
        # Train management
        'num_epochs': args.epochs,  # Number of training episodes
        'num_episodes': args.epochs,  # Number of training episodes
        'num_ppo_epochs': args.ppo_epochs,
        'mini_batch_size': args.ppo_batch_size,
        'memory_size': args.ppo_memory_size,
        'eps': args.ppo_epsilon,
        'c1': args.ppo_value_coeff,  # Value Function coeff
        'c2': args.ppo_entropy_coeff,  # Entropy coeff
        'lr': args.lr,  # Learning rate
        'gamma': args.gamma,  # Discount rate
        'stack_frames': 4,
        'action_set_num': args.action_set,
    }

    helpers.print_hyperparams(hyperparams)

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
        trainer = trainer_factory.build(hyperparams['strategy'], env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
