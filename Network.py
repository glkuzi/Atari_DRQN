import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from Config import FLICKERING, momentum, batch_size, real_batch_size


class DQN(nn.Module):
    def __init__(self, height, width, action_space):
        super(DQN, self).__init__()
        self.height = (((((height - 8) // 4 + 1) - 4) // 2 + 1) - 3) + 1
        self.width = (((((width - 8) // 4 + 1) - 4) // 2 + 1) - 3) + 1

        # layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)

        # layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # layer 4
        #self.lstm = nn.LSTMCell(self.height * self.width * 64, 512)
        self.lstm = nn.LSTM(self.height * self.width * 64, 512)
        # layer 5
        self.fc = nn.Linear(512, action_space)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hidden):
        #obs = 1
        #if FLICKERING:
        #    obs = random.randint(0, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        if x.shape[0] == 1:
            x = x.view(x.shape[0], 1, self.height * self.width * 64)
        else:
            x = x.reshape(batch_size - 1, x.shape[0] // (batch_size - 1),
                          self.height * self.width * 64)

        #x = x.view(x.shape[0], 1, self.height * self.width * 64)
        #print(x.shape)
        x, (h, c) = self.lstm(x, hidden)

        x = self.fc(x)
        #print('final shape', x.shape)
        return x, (h, c)


class DQN_Operator:
    def __init__(self, height, width, action_space, learning_rate, model_path = None):
        self.action_space = action_space

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.behaviourPolicy = DQN(height, width, action_space).to(self.device)
        self.targetPolicy = DQN(height, width, action_space).to(self.device)
        # use adadelta instead of adam, as in paper
        self.optimizer = optim.Adadelta(self.behaviourPolicy.parameters(), lr=learning_rate, rho=momentum)
        if model_path and os.path.isfile(model_path):
            net_parameter = torch.load(model_path)
            self.behaviourPolicy.load_state_dict(net_parameter)
            print('loaded model')
        self.update_targetPolicy()

    def update_targetPolicy(self):
        self.targetPolicy.load_state_dict(self.behaviourPolicy.state_dict())

    def train(self, s_batch, a_batch, r_batch, t_batch, gamma):
        """
        train behaviourPolicy

        input:
          s_batch       -- batch of current state s,                             shape of (batch size, 1, height, width)
          a_batch       -- batch of action in current state s,                   shape of (batch size, 1)
          r_batch       -- batch of rewad by action a at state s,                shape of (batch size, 1)
          t_batch       -- batch of whether the game has terminated at s_prime,  shape of (batch size, 1) 
          gamma         -- discount factor
        """

        s_batch = torch.FloatTensor(s_batch).to(self.device)

        hb, cb = torch.zeros([1, real_batch_size, 512], dtype=torch.float), torch.zeros([1, real_batch_size, 512], dtype=torch.float)
        ht, ct = torch.zeros([1, real_batch_size, 512], dtype=torch.float), torch.zeros([1, real_batch_size, 512], dtype=torch.float)
        hb, cb, ht, ct = hb.to(self.device), cb.to(self.device), ht.to(self.device), ct.to(self.device)
        loss = 0
        idx = 0
        # seq part
        for j in range(len(t_batch)):
            if 0 in t_batch[j]:
                # sequence has end of episode
                start = t_batch[j].index(0)
                for i in range(start, len(t_batch[j])):
                    t_batch[j][i] *= 0
        s = s_batch[:, :-1]
        s_prime = s_batch[:, 1:]
        r = torch.tensor(r_batch).to(self.device)
        #print('flatten', s.flatten(end_dim=1).shape, s.shape)
        Q_s, (hb, cb) = self.behaviourPolicy(s.flatten(end_dim=1), (hb, cb))
        t = torch.tensor(t_batch).to(self.device)
        a = torch.tensor(a_batch).to(self.device)
        #print(a[:, :-1])
        #print(Q_s.permute((1, 0, 2)).shape, a[:, :-1].flatten(end_dim=1).shape)
        #print(Q_s.permute((1, 0, 2)).flatten(end_dim=1).shape, t.shape)
        Q_a = torch.diagonal(torch.index_select(Q_s.permute((1, 0, 2)).flatten(end_dim=1), -1,
                                                a[:, :-1].flatten(end_dim=1))) * t[:, :-1].flatten()
        #Q_a = Q_s.gather(-1, torch.tensor(a_batch[:-1]).to(self.device)) * t_batch[:-1]#[:, :, a_batch[:-1]] * t_batch[:-1]
        Q_s_prime, (ht, ct) = self.targetPolicy(s_prime.flatten(end_dim=1), (ht, ct))
        Q_s_prime_max = Q_s_prime.max(-1)[0]
        #print(Q_s_prime_max.shape, r.shape, t.shape)
        target = ((r[:, :-1] + gamma * Q_s_prime_max.permute((1, 0))) * t[:, :-1]).flatten()
        #print(Q_a.shape, target.shape)
        # print(Q_a.shape, target.shape)
        loss = F.mse_loss(Q_a.squeeze(), target)#F.smooth_l1_loss(Q_a.squeeze(), target)
        '''
        while idx < len(a_batch)-1:
            if t_batch[idx] == 0.0:  # if the next state is terminate state
                s = s_batch[idx].unsqueeze(0)
                a = a_batch[idx]
                r = r_batch[idx]

                Q_s, (hb, cb) = self.behaviourPolicy(s, (hb, cb))
                Q_a = Q_s[:, a]

                target = r + torch.zeros_like(Q_a)

                loss += F.smooth_l1_loss(Q_a, target)
                idx += 1

                hb, cb = self.init_hidden()
                ht, ct = self.init_hidden()
                hb, cb, ht, ct = hb.to(self.device), cb.to(self.device), ht.to(self.device), ct.to(self.device)
                if idx == len(a_batch)-1:  # if the index is the last index, break because there is no next state in the batch
                    break

            s = s_batch[idx].unsqueeze(0)
            s_prime = s_batch[idx+1].unsqueeze(0)
            a = a_batch[idx]
            r = r_batch[idx]

            Q_s, (hb, cb) = self.behaviourPolicy(s, (hb, cb))
            Q_a = Q_s[:, a]

            Q_s_prime, (ht, ct) = self.targetPolicy(s_prime, (ht, ct))
            Q_s_prime_max = Q_s_prime.max(1)[0]
            target = r + gamma * Q_s_prime_max

            loss += F.smooth_l1_loss(Q_a, target)

            idx += 1
        '''
        self.optimizer.zero_grad()
        loss.backward()
        # clip lstm gradients to 10, as in paper
        torch.nn.utils.clip_grad_norm_(self.behaviourPolicy.lstm.parameters(), 10)
        self.optimizer.step()

    def action_epsilon_greedy(self, epsilon, state, hidden):
        """
        get action by epsilon greedy to Q

        input:
          epsilon -- exploration factor
          state   -- current state, numpy array of (height, width)
          hidden -- hidden of lstm layer (hidden state, cell state)
        """
        state = torch.FloatTensor([state]).to(self.device)#.unsqueeze(1)
        h, c = hidden
        h = h.to(self.device)
        c = c.to(self.device)
        output, (h, c) = self.behaviourPolicy(state, (h, c))
        # print(output.shape, output)
        # output = output[-1]
        # print(output, output.argmax().item())
        h = h.detach().cpu()
        c = c.detach().cpu()

        coin = random.random()
        if coin < epsilon:
            action = random.randint(0, self.action_space-1)
        else:
            action = output.argmax().item()

        return action, (h, c)

    def init_hidden(self):
        h, c = torch.zeros([1, 1, 512], dtype=torch.float), torch.zeros([1, 1, 512], dtype=torch.float) 
        return h, c

    def save(self, model_path):
        net_parameter = self.behaviourPolicy.state_dict()
        torch.save(net_parameter, model_path)
        print('saved model')
