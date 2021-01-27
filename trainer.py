import torch
import numpy as np
from policy import Policy
from actions import Actions
from helpers import save_model


class Trainer:

    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.gamma = config['gamma']
        self.config = config
        self.input_channels = 4
        self.policy = Policy(self.input_channels, len(Actions.available_actions))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])

    def select_action(self, state):
        if state is None:
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).view(1, self.input_channels, 96, 96)
        # Pick the probs of a discrete number of action (discrete mode not supported)
        probs = self.policy(state)
        action_index = torch.argmax(probs, 1)
        self.policy.saved_log_probs.append(probs[0][action_index])
        return Actions[action_index.item()]

    def episode_train(self):
        g = 0
        policy_loss = []
        returns = []

        for r in self.policy.rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns)
        # Normalize returns (this usually accelerates convergence)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, G in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-G * log_prob)

        # Update policy:
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        print(len(self.policy.rewards))
        print(len(self.policy.saved_log_probs))
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        running_reward = 10
        ep_rew_history = []
        for i_episode in range(self.config['num_episodes']):
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
            self.episode_train()
            ep_rew_history.append((i_episode, ep_reward))
            if i_episode % self.config['log_interval'] == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                save_model(self.policy, './params/policy-params.dl')

            if running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
