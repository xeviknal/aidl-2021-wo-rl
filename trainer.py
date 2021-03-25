import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections import namedtuple


from policy import Policy
from actions import available_actions
from memory import ReplayMemory, Transition


class Trainer:

    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.gamma = config['gamma']
        self.input_channels = config['stack_frames']
        self.device = config['device']
        self.epochs = config['num_epochs']
        self.mini_batch = config['batch_size']
        self.c1, self.c2 = config['c1'], config['c2']
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
        return action.item(), m.log_prob(action), vs_t, m.entropy()

    def run_episode(self, current_steps):
        state, ep_reward, steps = self.env.reset(), 0, 0
        for t in range(self.env.spec().max_episode_steps):  # Protecting from scenarios where you are mostly stopped
            action_id, action_log_prob, vs_t, entropy = self.select_action(state)
            next_state, reward, done, _ = self.env.step(available_actions[action_id])
            # Store transition to memory
            self.memory.push(state, action_id, action_log_prob, entropy, reward, vs_t, next_state)

            steps += 1
            current_steps += 1
            ep_reward += reward
            # TODO: memory size
            if done or current_steps == 2000:
                break

        # Update running reward
        self.running_reward = 0.05 * ep_reward + (1 - 0.05) * self.running_reward
        self.logging_episode(i_episode, ep_reward)
        return steps

    def policy_update(self, transitions):
        # Get transitions values
        batch = ReplayMemory.Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        log_prop_batch = torch.cat(batch.log_prob)
        entropy_batch = torch.cat(batch.entropy)
        next_state_batch = torch.cat(batch.next_step)

        # TODO: need to apply the mask for last states of the episode?
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.policy(non_final_next_states).max(dim=1)[0].detach()

        l_clip = 0 # good luck
        with torch.no_grad:
            _, v_t1 = self.policy(next_state_batch)
            v_targ = reward_batch + self.gamma * v_t1

        # TODO: k epochs and transitions loop
        l_vp = nn.SmoothL1Loss(v_theta - v_targ)
        l_entropy = self.c2 * entropy_batch

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() + torch.stack(value_losses).sum()
        self.writer.add_scalar('loss', policy_loss.item(), iteration)
        policy_loss.backward()
        self.optimizer.step()

    def clean_training_batch(self):
        self.memory.clear()  # Clearing memory
        del self.policy.saved_log_probs[:]
        del self.policy.rewards[:]

    def logging_episode(self, i_episode, ep_reward):
        self.writer.add_scalar('reward', ep_reward, i_episode)
        self.writer.add_scalar('running reward', self.running_reward, i_episode)
        # self.writer.add_scalar('mean entropy', np.mean(self.policy.entropies), i_episode)
        # self.writer.add_scalar('mean action prob',
        #                       torch.mean(torch.exp(torch.Tensor(self.policy.saved_log_probs)[:, :1])), i_episode)

    def train(self):
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        # TODO: When do we finish?
        for i_episode in range(self.config['num_episodes'] - self.last_episode):
            # Convert to 1-indexing to reduce complexity
            i_episode += 1
            # The episode counting starts from last checkpoint
            i_episode = i_episode + self.last_episode

            # Collect experience // Filling the memory (2000 positions)
            # TODO: loop until we get 2k transitions in the memory
            # TODO: hyperparam for memory size
            steps = 0
            while steps <= 2000:
                steps += self.run_episode(steps)

            # Train the model num_epochs time with mini-batch strategy
            for i in range(self.epochs):
                # Train the model with batch-size transitions
                # TODO: replace 2k for memory size
                self.memory.shuffle()
                for k in range(0, 2000, self.mini_batch):
                    self.policy_update(self.memory.get_batch())

            # TODO: Is this necessary? Memory is rounded
            self.clean_training_batch()

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
