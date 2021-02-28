from trainers.trainer import Trainer


class BasicTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)

    def train(self):
        print("Target reward: {}".format(self.env.spec().reward_threshold))
        running_reward = 10
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
            self.writer.add_scalar('reward', ep_reward, i_episode)
            self.writer.add_scalar('running reward', running_reward, i_episode)
            if i_episode % self.config['log_interval'] == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                self.policy.save_checkpoint(self.config['params_path'], i_episode)

            if running_reward > self.env.spec().reward_threshold:
                print("Solved!")
                break

        print("Finished training! Running reward is now {:.2f}".format(running_reward))
