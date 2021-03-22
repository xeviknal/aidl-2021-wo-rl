import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import namedtuple


from policy import Policy
from actions import available_actions
from memory import ReplayMemory, Transition


class Trainer:

    def __init__(self, env, config):
        super().__init__()
        self.MarkovDecisionProcess = namedtuple('MarkovDecision', ['state', 'action', 'log_prob', 'vs_t', 'entropy'])
        self.env = env
        self.config = config
        self.gamma = config['gamma']
        self.input_channels = config['stack_frames']
        self.device = config['device']
        self.epochs = config['num_epochs']
        self.batch = config['batch_size']
        self.writer = SummaryWriter(flush_secs=5)
        self.policy = Policy(len(available_actions), 1, self.input_channels).to(self.device)
        self.last_episode, optim_params, self.running_reward = self.policy.load_checkpoint(config['params_path'])
        self.memory = ReplayMemory(2000)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])
        if optim_params is not None:
            self.optimizer.load_state_dict(optim_params)

    def select_action(self, state):
        if state is None:  # First state is always None
            # Adding the starting signal as a 0's tensor
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).view(1, self.input_channels, 96, 96).to(self.device)
        probs, vs_t = self.policy(state)
        # vs_t = return estimated for the current state

        # We pick the action from a sample of the probabilities
        # It prevents the model from picking always the same action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.policy.saved_current_mdp.append(self.MarkovDecisionProcess(state, action.item(), m.log_prob(action), vs_t, m.entropy()))
        return available_actions[action.item()]

    def run_episode(self):
        state, ep_reward = self.env.reset(), 0
        for t in range(self.env.spec().max_episode_steps):  # Protecting from scenarios where you are mostly stopped
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            self.policy.rewards.append(reward)

            ep_reward += reward
            if done:
                break

        return ep_reward

    def policy_update(self, iteration):
        g = 0
        policy_loss = []
        value_losses = []
        returns = []

        for r in self.policy.rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns).to(self.device)
        # Normalize returns (this usually accelerates convergence)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, baseline), G in zip(self.policy.saved_log_probs, returns):
            advantage = G - baseline.item()

            # calculate actor (policy) loss
            policy_loss.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(baseline.squeeze(), G))

        # Update policy:
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() + torch.stack(value_losses).sum()
        self.writer.add_scalar('loss', policy_loss.item(), iteration)
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def logging_episode(self, i_episode, ep_reward):
        self.writer.add_scalar('reward', ep_reward, i_episode)
        self.writer.add_scalar('running reward', self.running_reward, i_episode)
        # self.writer.add_scalar('mean entropy', np.mean(self.policy.entropies), i_episode)
        # self.writer.add_scalar('mean action prob',
        #                       torch.mean(torch.exp(torch.Tensor(self.policy.saved_log_probs)[:, :1])), i_episode)

    def train(self):
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        for i_episode in range(self.config['num_episodes'] - self.last_episode):
            # Convert to 1-indexing to reduce complexity
            i_episode += 1
            # The episode counting starts from last checkpoint
            i_episode = i_episode + self.last_episode

            # Collect experience // Filling the memory (2000 positions)
            for i in range(2):
                ep_reward = self.run_episode()
                # Update running reward
                self.running_reward = 0.05 * ep_reward + (1 - 0.05) * self.running_reward
                self.logging_episode(i_episode, ep_reward)

            for k in range(self.epochs):
                for b in range(self.batch):
                    self.episode_train(i_episode)

            # Saving each log interval, at the end of the episodes or when training is complete
            # TODO: catch keyboard interrupt
            if i_episode % self.config['log_interval'] == 0 or i_episode == self.config['num_episodes'] or self.running_reward > self.env.spec().reward_threshold:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, self.running_reward))
                self.policy.save_checkpoint(self.config['params_path'], i_episode, self.running_reward, self.optimizer)

            if self.running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f}".format(self.running_reward))
