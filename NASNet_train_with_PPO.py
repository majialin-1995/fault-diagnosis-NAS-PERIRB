import torch
import configparser
import copy
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from NASSNet.env import Environment
from Datasets.CWRU import CWRU
from core.utils import seed_everything
import torch.nn.utils.rnn as rnn_utils
import os

import argparse
import pickle
from collections import namedtuple
from itertools import count

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

config_file = 'config/CWRU_TYPE4_NASSNet.ini'
config = configparser.ConfigParser()
config.read(config_file)
process_config = config['PROCESS']
config_net_training = config['NET_TRAINING']
config_rl_training = config['RL_TRAINING']
seed_everything(process_config.getint('random_seed'))
###################### Environment #################
batch_size = config_net_training.getint('batch_size')
datasets = CWRU(config=process_config)
x_train = torch.from_numpy(datasets.X_train).type(torch.FloatTensor)
x_test = torch.from_numpy(datasets.X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(datasets.y_train).type(torch.LongTensor)
y_test = torch.from_numpy(datasets.y_test).type(torch.LongTensor)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
env = Environment(config, train_dataloader, test_dataloader)
#####################################################
###################TRAIN PROCESS#####################
#####################################################


# Parameters
gamma = 0.99
render = False
log_interval = 10

Transition = namedtuple(
    'Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=n_s, hidden_size=n_s, num_layers=1)
        self.fc1 = nn.Linear(n_s, 32)
        self.fc2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, n_a)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        x = self.fc1(h.squeeze(0))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, n_s):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=n_s, hidden_size=n_s, num_layers=1)
        self.fc1 = nn.Linear(n_s, 32)
        self.fc2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        x = self.fc1(h.squeeze(0))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.out(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 2

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor(13, 13)
        self.critic_net = Critic(13)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(1)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(
        ), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(
        ), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # Add Done action
        bufferlength = len(self.buffer)
        for i in range(bufferlength):
            sample = copy.copy(self.buffer[i])
            new_sample = Transition(
                sample.state, 0, sample.a_log_prob, sample.reward, None)
            self.buffer.insert(i*2+1, new_sample)

        state = [t.state for t in self.buffer]

        action = torch.tensor(
            [t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state  for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        # This is for Insert Replay Buffer

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)

        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(
                        i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)

                # now state is a list and irregular
                # train_state = state[index] is not a right way
                train_state = [state[i] for i in index]
                x_length = [len(sq) for sq in train_state]
                train_state = rnn_utils.pad_sequence(
                    train_state, batch_first=True)
                train_state = rnn_utils.pack_padded_sequence(
                    train_state, x_length, batch_first=True, enforce_sorted=False)

                V = self.critic_net(train_state)
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(train_state).gather(
                    1, action[index])  # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                    1 + self.clip_param) * advantage

                # update actor network
                # MAX->MIN desent
                action_loss = -torch.min(surr1, surr2).mean()
                self.writer.add_scalar(
                    'loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar(
                    'loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        del self.buffer[:]  # clear experience


def main():
    net_datas = []
    agent = PPO()
    for i_epoch in range(1000):
        state, done, info = env.reset()
        if render:
            env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trans = Transition(torch.tensor(
                state, dtype=torch.float), action, action_prob, reward, next_state)
            if render:
                env.render()
            agent.store_transition(trans)
            state = next_state

            net_datas.append({
                "state": copy.copy(env.net_structures),
                "reward": reward,
                "acc": env.acc
            })

            pd_data = pd.DataFrame(net_datas)
            pd_data.to_csv('net_datas_PPO.csv', index=False)
            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                agent.writer.add_scalar(
                    'liveTime/livestep', t, global_step=i_epoch)
                break


if __name__ == '__main__':
    main()
    print("end")
