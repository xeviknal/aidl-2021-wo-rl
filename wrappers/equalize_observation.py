import gym
import torch
import PIL.ImageOps as ops
from PIL import Image
import numpy as np

__all__ = ['EqualizeObservation']


class EqualizeObservation(gym.ObservationWrapper):
    """
        Equalizing observations at each step
    """

    def __init__(self, env):
        super(EqualizeObservation, self).__init__(env)

    def observation(self, observation):
        img = Image.fromarray(observation)
        eq_img = ops.equalize(img)
        return np.asarray(eq_img)
