import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from actions import get_action
from memory import ReplayMemory, Transition
from policy import Policy


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
        self.action_set = get_action(config['action_set_num'])
        self.policy = Policy(len(self.action_set), 1, self.input_channels).to(self.device)
        self.last_epoch, optim_params, self.running_reward = self.policy.load_checkpoint(config['params_path'])
        self.memory = ReplayMemory(self.memory_size)
        self.value_loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.experiment = config['experiment']
        if optim_params is not None:
            self.optimizer.load_state_dict(optim_params)

    def prepare_state(self, state):
        if state is None:  # First state is always None
            # Adding the starting signal as a 0's tensor
            state = np.zeros((self.input_channels, 96, 96))
        else:
            state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).view(1, self.input_channels, 96, 96).to(self.device)
        return state

    def select_action(self, state):
        probs, vs_t = self.policy(state)
        # vs_t = return estimated for the current state

        # We pick the action from a sample of the probabilities
        # It prevents the model from picking always the same action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        # We return the state in order to make sure that we operate with a valid tensor
        return action, m.log_prob(action), vs_t, m.entropy(), state

    def run_episode(self, epoch, current_steps):
        state, ep_reward, steps = self.env.reset(), 0, 0
        for t in range(self.env.spec().max_episode_steps):  # Protecting from scenarios where you are mostly stopped
            with torch.no_grad():
                state = self.prepare_state(state)
                action_id, action_log_prob, vs_t, entropy, state = self.select_action(state)
                next_state, reward, done, _ = self.env.step(self.action_set[action_id.item()])
                # Store transition to memory
                self.memory.push(state, action_id, action_log_prob, entropy, reward, vs_t, next_state)

                state = next_state
                steps += 1
                current_steps += 1
                ep_reward += reward
                if done or current_steps == self.memory_size:
                    break

        # Update running reward
        if done:
            self.running_reward = 0.05 * ep_reward + (1 - 0.05) * self.running_reward

        self.logging_episode(epoch, ep_reward, self.running_reward)
        return steps

    def compute_advantages(self):
        batch = Transition(*zip(*self.memory.memory))
        s = torch.cat(batch.state).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).view(-1, 1).to(self.device)
        next_state_batch = torch.from_numpy(np.asarray(batch.next_state)).float().unsqueeze(0).view(len(self.memory), self.input_channels, 96, 96).to(self.device)

        with torch.no_grad():
            # Computing expected future return for t+1 step: Gt+1
            v_targ = reward_batch + self.gamma * self.policy(next_state_batch)[1]
            # Computing advantage
            adv = v_targ - self.policy(s)[1]

        return v_targ, adv

    def policy_update(self, transitions, v_targ, adv, iteration):
        # Get transitions values
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        old_log_prop_batch = torch.cat(batch.log_prob)
        entropy_batch = torch.cat(batch.entropy).view(-1, 1)
        vst_batch = torch.cat(batch.vs_t)

        l_vf = self.c1 * self.value_loss(vst_batch, v_targ)
        l_entropy = self.c2 * entropy_batch.mean()

        #  Computing clipped loss:
        _, new_log_prob_batch, _, _, _ = self.select_action(state_batch)

        # For performance reasons. rt = exp(new_log_prob) / exp(old_log_prop)
        rt = torch.exp(new_log_prob_batch - old_log_prop_batch)
        clipped = adv * torch.clip(rt, 1.0 - self.eps, 1.0 + self.eps)
        # We apply the mean because we want to compute the expected value
        l_clip = torch.min(rt * adv, clipped).mean()

        # loss definition = lclip - l_vf + l_entropy
        # We want to maximise the prob of optimal action
        # but SGD looks for the minimum. Therefore, we need
        # to invert the sign of the loss.
        loss = -l_clip + l_vf - l_entropy

        self.optimizer.zero_grad()
        self.writer.add_scalar(f'{self.experiment}/loss', loss.item(), iteration)
        self.writer.add_scalar(f'{self.experiment}/entropy', l_entropy.item(), iteration)
        self.writer.add_scalar(f'{self.experiment}/ratio', rt.mean().item(), iteration)
        self.writer.add_scalar(f'{self.experiment}/advantage', adv.mean().item(), iteration)
        self.writer.add_scalar(f'{self.experiment}/vf', l_vf.item(), iteration)
        loss.backward()
        self.optimizer.step()

    def logging_episode(self, i_episode, ep_reward, running_reward):
        self.writer.add_scalar(f'{self.experiment}/reward', ep_reward, i_episode)
        self.writer.add_scalar(f'{self.experiment}/running reward', running_reward, i_episode)

    def train(self):
        # Training loop
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        global_step = 0
        for epoch in range(self.epochs - self.last_epoch - 1):
            # The episode counting starts from last checkpoint
            epoch += self.last_epoch

            # Collect experience // Filling the memory (self.memory_size positions)
            steps, ep_count = 0, 0
            while steps < self.memory_size:
                ep_steps = self.run_episode(epoch, steps)
                steps += ep_steps
                ep_count += 1

            v_targ, adv = self.compute_advantages()

            # Train the model num_epochs time with mini-batch strategy
            for ppo_epoch in range(self.ppo_epochs):
                # Train the model with batch-size transitions
                for index in BatchSampler(SubsetRandomSampler(range(self.memory_size)), self.mini_batch, False):
                    self.policy_update(self.memory[index], v_targ[index], adv[index], global_step)
                    global_step += 1

            # Saving each log interval, at the end of the episodes or when training is complete
            # TODO: catch keyboard interrupt
            if epoch % self.config['log_interval'] == 0 or epoch == self.epochs \
                    or self.running_reward > self.env.spec().reward_threshold:
                print('Epoch {}\t Average reward: {:.2f}'.format(epoch, self.running_reward))
                self.policy.save_checkpoint(self.config['params_path'], epoch, self.running_reward, self.optimizer)

            if self.running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f}".format(self.running_reward))
