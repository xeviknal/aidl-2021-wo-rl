import torch

from environment import CarRacingEnv
from trainer import Trainer

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hyperparams = {
        'num_episodes': 30000,  # Number of training episodes
        'lr': 1e-2,  # Learning rate
        'gamma': 0.99,  # Discount rate
        'log_interval': 5,  # controls how often we log progress
        'stack_frames': 4,
        'device': device,
        'params_path': './params/policy-params.dl'
    }

    env = CarRacingEnv(device, hyperparams['stack_frames'])
    trainer = Trainer(env, hyperparams)
    trainer.train()

