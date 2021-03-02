import torch

import helpers
from environment import CarRacingEnv
from trainers.curriculum import CurriculumTrainer
from runner import Runner

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
        'train': True,
        # Curriculum learning params
        'curriculum_step_length': 100,
        'curriculum_speed': 4.0,
    }

    env = CarRacingEnv(device, hyperparams['stack_frames'], hyperparams['train'])
    helpers.display_start()
    if hyperparams['train']:
        trainer = CurriculumTrainer(env, hyperparams)
        trainer.train()
    else:
        runner = Runner(env, hyperparams)
        runner.run()
        

