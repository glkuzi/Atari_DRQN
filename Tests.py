import time
import random
from Network import DQN_Operator
from Environment import atari_env
from Config import *
import cv2


env = atari_env(preprocess_height, preprocess_width, name)
s = env.reset()
act_space = env.env.action_space.n
for i in range(100):
    a = random.randint(0, act_space-1)
    s_prime, r, terminate, info = env.step(a)
    # print(s_prime[0].shape)
    cv2.imshow("normGray", s_prime[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()