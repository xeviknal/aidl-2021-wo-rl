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
        self.ppo_epochs = config['num_ppo_epochs']
        self.mini_batch = config['mini_batch_size']
        self.memory_size = config['memory_size']
        self.c1, self.c2, self.eps = config['c1'], config['c2'], config['eps']
        self.writer = SummaryWriter(flush_secs=5)
        self.policy = Policy(len(available_actions), 1, self.input_channels).to(self.device)
        self.last_epoch, optim_params, self.running_reward = self.policy.load_checkpoint(config['params_path'])
        self.memory = ReplayMemory(self.memory_size, self.mini_batch)
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

            state = next_state
            steps += 1
            current_steps += 1
            ep_reward += reward
            if done or current_steps == self.memory_size:
                break

        # Update running reward
        self.running_reward = 0.05 * ep_reward + (1 - 0.05) * self.running_reward
        return steps, ep_reward

    def policy_update(self, transitions, iteration):
        # Get transitions values
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        old_log_prop_batch = torch.cat(batch.log_prob)
        entropy_batch = torch.cat(batch.entropy)
        reward_batch = torch.cat(batch.reward)
        vst_batch = torch.cat(batch.vs_t)
        next_state_batch = torch.cat(batch.next_step)

        # TODO: need to apply the mask for last states of the episode?
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.policy(non_final_next_states).max(dim=1)[0].detach()

        with torch.no_grad:
            # Computing expected future return for t+1 step: Gt+1
            _, v_t1 = self.policy(next_state_batch)
            v_targ = reward_batch + self.gamma * v_t1
            # Computing advantage
            adv = vst_batch - v_targ

        l_vf = self.c1 * nn.SmoothL1Loss(vst_batch, v_targ)
        l_entropy = self.c2 * entropy_batch

        #  Computing clipped loss:
        _, new_log_prob_batch, _, _ = self.select_action(state_batch)
        rt = torch.exp(new_log_prob_batch) / torch.exp(old_log_prop_batch)
        clipped = adv * torch.clip(rt, 1.0 - self.eps, 1.0 + self.eps)
        lclip = torch.min(rt * adv, clipped)

        loss = lclip - l_vf + l_entropy

        self.optimizer.zero_grad()
        self.writer.add_scalar('loss', loss.item(), iteration)
        loss.backward()
        self.optimizer.step()

    def clean_training_batch(self):
        self.memory.clear()  # Clearing memory
        del self.policy.saved_log_probs[:]
        del self.policy.rewards[:]

    def logging_episode(self, i_episode, ep_reward, running_reward):
        self.writer.add_scalar('reward', ep_reward, i_episode)
        self.writer.add_scalar('running reward', running_reward, i_episode)
        # self.writer.add_scalar('mean entropy', np.mean(self.policy.entropies), i_episode)
        # self.writer.add_scalar('mean action prob',
        #                       torch.mean(torch.exp(torch.Tensor(self.policy.saved_log_probs)[:, :1])), i_episode)

    def train(self):
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        for epoch in range(self.epochs - self.last_epoch - 1):
            # The episode counting starts from last checkpoint
            epoch += self.last_epoch

            # Collect experience // Filling the memory (self.memory_size positions)
            steps, ep_reward, ep_count = 0, 0, 0
            while steps < self.memory_size:
                ep_steps, ep_reward = self.run_episode(steps)
                steps += ep_steps
                ep_count += 1

            # Train the model num_epochs time with mini-batch strategy
            for ppo_epoch in range(self.ppo_epochs):
                # Train the model with batch-size transitions
                self.memory.shuffle()
                for k in range(0, self.memory_size, self.mini_batch):
                    # TODO: define number of update
                    self.policy_update(self.memory.get_batch(), 100)

            # TODO: Is this necessary? Memory is rounded
            self.clean_training_batch()

            self.logging_episode(self.epochs, ep_reward, self.running_reward)

            # Saving each log interval, at the end of the episodes or when training is complete
            # TODO: catch keyboard interrupt
            if epoch % self.config['log_interval'] == 0 or epoch == self.epochs \
                    or self.running_reward > self.env.spec().reward_threshold:
                print('Epoch {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    epoch, ep_reward, self.running_reward))
                self.policy.save_checkpoint(self.config['params_path'], epoch, self.running_reward, self.optimizer)

            if self.running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f}".format(self.running_reward))
