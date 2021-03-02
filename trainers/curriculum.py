from trainers.trainer import Trainer
from trainers.basic import BasicTrainer


class CurriculumTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(CurriculumTrainer, self).__init__(*args, **kwargs)
        self.speed = self.config["curriculum_speed"]
        self.max_episode_step = self.env.spec().max_episode_steps
        self.curriculum_range_length = self.config["curriculum_step_length"]

    def curriculum_ranges(self):
        ranges = self.max_episode_step // self.curriculum_range_length
        for i in range(ranges):
            steps = (i + 1) * self.curriculum_range_length
            if i == ranges - 1 and steps < self.max_episode_step:
                steps = self.max_episode_step
            yield steps, ranges

    def reward_goal_for(self, range_length):
        return range_length - (range_length * self.env.step_reward * self.speed)

    def train(self):
        running_reward = 10
        for curriculum_range, range_count in self.curriculum_ranges():
            reward_goal = self.reward_goal_for(curriculum_range)
            max_episodes_per_range = self.config['num_episodes'] // range_count
            print("Curriculum range: {}".format(curriculum_range, reward_goal))
            basic_training = BasicTrainer(self.env, self.config, curriculum_range, reward_goal, max_episodes_per_range)
            running_reward = basic_training.train()
            print("Finished training! Running reward is now {:.2f}".format(running_reward))

        if running_reward >= self.env.spec().reward_threshold:
            print("Solved with a running reward of {:.2f}".format(running_reward))
