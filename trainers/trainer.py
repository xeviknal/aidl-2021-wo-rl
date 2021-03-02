import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from policy import Policy
from actions import available_actions


class Trainer:

    def __init__(self, env, config, episode_length=1000, reward_goal=900, max_epochs=1000):
        super().__init__()
        self.env = env
        self.config = config
        self.episode_length = episode_length
        self.reward_goal = reward_goal
        self.max_epochs = max_epochs
        self.gamma = config['gamma']
        self.input_channels = config['stack_frames']
        self.device = config['device']
        self.writer = SummaryWriter(flush_secs=5)
        self.policy = Policy(self.input_channels, len(available_actions)).to(self.device)
        self.last_epoch = self.policy.load_checkpoint(config['params_path']) | 0
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])

    def select_action(self, state):
        if state is None:  # First state is always None
            # Adding the starting signal as a 0's tensor
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        # We pick the action from a sample of the probabilities
        # It prevents the model from picking always the same action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return available_actions[action.item()]

    def episode_train(self, iteration):
        g = 0
        policy_loss = []
        returns = []

        for r in self.policy.rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns).to(self.device)
        # Normalize returns (this usually accelerates convergence)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, G in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-G * log_prob)

        # Update policy:
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        self.writer.add_scalar('loss', policy_loss.item(), iteration)
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        raise NotImplementedError
