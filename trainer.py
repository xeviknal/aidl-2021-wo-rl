import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from policy import Policy
from actions import Actions


class Trainer:

    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.gamma = config['gamma']
        self.config = config
        self.input_channels = config['stack_frames']
        self.device = config['device']
        self.writer = SummaryWriter(flush_secs=5)
        self.policy = Policy(self.input_channels, len(Actions.available_actions)).to(self.device)
        self.last_epoch = self.policy.load_checkpoint(config['params_path'])
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
        return Actions[action.item()]

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
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        running_reward = 10
        ep_rew_history = []
        for i_episode in range(self.config['num_episodes'] - self.last_epoch):
            # The episode counting starts from last checkpoint
            i_episode = i_episode + self.last_epoch
            # Collect experience
            state, ep_reward = self.env.reset(), 0
            for t in range(self.env.spec().max_episode_steps):  # Protecting from scenarios where you are mostly stopped
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                self.policy.rewards.append(reward)

                ep_reward += reward
                if done:
                    break

            # Update running reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # Perform training step
            self.episode_train(i_episode)
            ep_rew_history.append((i_episode, ep_reward))
            self.writer.add_scalar('reward', ep_reward, i_episode)
            if i_episode % self.config['log_interval'] == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                self.policy.save_checkpoint('./params/policy-params.dl', i_episode)

            if running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f}".format(running_reward))
