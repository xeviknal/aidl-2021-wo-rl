from trainers.trainer import Trainer


class CurriculumTrainer(Trainer):

    def __init__(self, speed=1, *args, **kwargs):
        super(CurriculumTrainer, self).__init__(*args, **kwargs)
        self.speed = speed

    def curriculum_ranges(self):
        max_episode_step = self.env.spec().max_episode_steps
        curriculum_range_length = self.config["curriculum_step_length"]
        ranges = max_episode_step // curriculum_range_length
        # 1000 / 100 -> 10 / 100 - (0.1 * 100)
        # 1000 / 150 -> 7 (less 50 steps) - (0.1 * 150)
        for i in range(ranges):
            steps = (i + 1) * curriculum_range_length
            if i == ranges - 1 and steps < max_episode_step:
                steps = max_episode_step
            yield steps

    def train(self):
        running_reward = 10

        for curriculum_range in self.curriculum_ranges():
            print(curriculum_range)

