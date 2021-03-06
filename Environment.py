import gym
import random
import numpy as np
import cv2
from Config import FLICKERING


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.Buffer = []
        self.position = 0

    def push(self, transition):
        """
        push transition data to Beffer

        input:
          transition -- list of [s, a, r, t]
        """
        if len(self.Buffer) < self.capacity:
            self.Buffer.append(None)
        self.Buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random_idx = random.randint(0, len(self.Buffer) - batch_size)
        mini_batch = self.Buffer[random_idx: random_idx + batch_size]

        s_batch, a_batch, r_batch, t_batch = [], [], [], []
        for transition in mini_batch:
            s, a, r, t = transition

            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            t_batch.append(t)

        return s_batch, a_batch, r_batch, t_batch

    def size(self):
        return len(self.Buffer)


class atari_env:
    def __init__(self, height, width, name):
        self.env = gym.make(name)
        self.height = height
        self.width = width

    def pre_process(self, s):
        """
        preprocessing the given image, gray scailing and change it to 84 x 84

        input:
          s -- given state, RGB image of 210 * 160 *3
        """
        #cv2.imshow("RGB", s)
        s = cv2.cvtColor(cv2.resize(s, (self.height, self.width)),
                         cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray", s)
        s = np.divide(s, 255.0)
        #cv2.imshow("normGray", s)
        s = np.expand_dims(s, axis=0)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        obs = 1
        if FLICKERING:
            obs = random.randint(0, 1)
        return s * obs / 255.0

    def step(self, a):
        s_prime, r, terminate, info = self.env.step(a)
        s_prime = self.pre_process(s_prime)
        return s_prime, r, terminate, info

    def reset(self):
        s = self.env.reset()
        s = self.pre_process(s)
        return s

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
