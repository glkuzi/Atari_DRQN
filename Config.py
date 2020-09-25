# -*- coding: utf-8 -*-

train_episode_num = 3000
learning_rate = 0.1
momentum = 0.95
gamma = 0.99
buffer_capacity = 400000
batch_size = 10
replay_start_size = 50000
final_exploration_step = 1000000
update_interval = 10000  # target net
update_frequency = 1  # the number of actions selected by the agent between successive SGD updates
preprocess_height = 84
preprocess_width = 84
name = 'Pong-v0'
model_path = './DRQN.model'
start_step = 0  # 12039628
start_ep = 0  # 15820
play_epi_num = 100
# if True, observation is 0 with 0.5 probability
FLICKERING = True
