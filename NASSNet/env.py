import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

'''
定义动作
0: Done
1: Conv1d(channels 2 times, kernel_size=1)
2: Conv1d(channels 2 times, kernel_size=3)
3: Conv1d(channels 2 times, kernel_size=5)
4: Conv1d(channels 1/2 times, kernel_size=1)
5: Conv1d(channels 1/2 times, kernel_size=3)
6: Conv1d(channels 1/2 times, kernel_size=5)
7: MaxPool1d(kernel_size=3)
8: MaxPool1d(kernel_size=5)
9: Dropout(0.5)
10: BatchNormalization
11: ReLU
'''

use_cuda = True
device = torch.device("cuda" if (
    use_cuda and torch.cuda.is_available()) else "cpu")


class Target_Net(nn.Module):
    def __init__(self, net_structures):
        super(Target_Net, self).__init__()
        self.net_structures = net_structures
        self.signal_length = 400
        self.in_channels = 1

        model = []
        for index, net_structure in enumerate(self.net_structures):
            if net_structure == 0:
                break
            if net_structure == 1:
                out_channels = self.in_channels * 2
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=1))
                self.in_channels = out_channels
            if net_structure == 2:
                out_channels = self.in_channels * 2
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=3))
                self.signal_length -= 2
                self.in_channels = out_channels
            if net_structure == 3:
                out_channels = self.in_channels * 2
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=5))
                self.signal_length -= 4
                self.in_channels = out_channels
            if net_structure == 4:
                out_channels = self.in_channels // 2
                out_channels = out_channels if out_channels > 0 else 1
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=1))
                self.in_channels = out_channels
            if net_structure == 5:
                out_channels = self.in_channels // 2
                out_channels = out_channels if out_channels > 0 else 1
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=3))
                self.signal_length -= 2
                self.in_channels = out_channels
            if net_structure == 6:
                out_channels = self.in_channels // 2
                out_channels = out_channels if out_channels > 0 else 1
                model.append(nn.Conv1d(
                    in_channels=self.in_channels, out_channels=out_channels, kernel_size=5))
                self.signal_length -= 4
                self.in_channels = out_channels
            if net_structure == 7:
                model.append(nn.MaxPool1d(kernel_size=3))
                self.signal_length = self.signal_length // 3
            if net_structure == 8:
                model.append(nn.MaxPool1d(kernel_size=5))
                self.signal_length = self.signal_length // 5
            if net_structure == 9:
                model.append(nn.Dropout(0.5))
            if net_structure == 10:
                model.append(nn.BatchNorm1d(self.in_channels))
            if net_structure == 11:
                model.append(nn.ReLU())
        self.last_layer = nn.Linear(self.signal_length * self.in_channels, 4)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0],-1)
        x = self.last_layer(x)
        x = F.softmax(x, dim=1)
        return x

class Environment:
    def __init__(self, config, train_dataloader, test_dataloader):
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.acc = 0
        self.layer_size = 0
        self.signal_length = 0

        self.net_structures = []

    def reset(self):
        '''声明一个最简单的网络
        '''
        self.net_structures = []
        self.net_structures.append(1)
        self.acc = 0
        self.layer_size = 0
        self.signal_length = 400
        # 初始化acc
        self._get_reward()
        self.max_acc = 0.0

        return self._get_state(), False, self._get_info()

    def step(self, action):
        '''动作定义为选择的网络结构
        添加到网络结构中
        '''
        self.net_structures.append(action)
        dis_acc, signal_length = self._get_reward()
        self.signal_length = signal_length
        return self._get_state(), dis_acc, (action == 0) or (signal_length < 5), self._get_info()

    def _get_state(self):
        '''获得环境的状态
        '''
        return np.eye(13)[copy.copy(self.net_structures)]

    def _get_info(self):
        net_structures_final = copy.copy(self.net_structures)
        net_structures_final.append(0)
        return np.eye(13)[net_structures_final]

    def _get_reward(self):
        '''训练，反馈识别率
        '''
        config_net_training = self.config['NET_TRAINING']
        target_model = Target_Net(net_structures=self.net_structures).to(device)

        opt_model = torch.optim.Adam(
            target_model.parameters(), lr=config_net_training.getfloat("lr"))

        epochs = config_net_training.getint('epochs')

        best_loss = 999

        valid_losses = []
        for epoch in range(epochs):
            target_model.train()
            print("{}/{}".format(epoch, epochs), end='\r')

            for i, data in enumerate(self.train_dataloader, 0):
                opt_model.zero_grad()
                train_signals, train_labels = data
                train_signals, train_labels = train_signals.to(
                    device), train_labels.to(device)

                logits_model = target_model(train_signals)
                loss_model = F.cross_entropy(logits_model, train_labels)
                loss_model.backward()
                opt_model.step()

            target_model.eval()

            for i, data in enumerate(self.test_dataloader, 0):
                logits_model = target_model(train_signals)
                loss = F.cross_entropy(logits_model, train_labels)
                valid_losses.append(loss.item())
            
            valid_loss = np.average(valid_losses)
            
            # This is for early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                es = 0
            else:
                es += 1
                if es > 4:
                    break

        num_correct = 0
        num_total = 0
        for i, data in enumerate(self.test_dataloader, 0):
            test_signal, test_label = data
            test_signal, test_label = test_signal.to(
                device), test_label.to(device)
            pred_lab = torch.argmax(target_model(test_signal), 1)
            num_correct += torch.sum(pred_lab == test_label, 0)
            num_total += len(test_label)

        new_acc = (float)(num_correct.item()/num_total)
        dis_acc = new_acc - self.acc
        self.acc = new_acc

        # This is for PER
        if self.acc > self.max_acc:
            self.max_acc = self.acc
            dis_acc = self.acc - self.max_acc * (.9 ** target_model.signal_length)

        return dis_acc, target_model.signal_length


