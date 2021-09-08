import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_signal(signal):
    plt.plot(signal)
    plt.show()


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def constant_wgns(x_datas, noise_power):

    def wgn(x, noise_power):
        return np.random.randn(len(x)) * np.sqrt(noise_power)

    results_datas = []
    for x_data in x_datas:
        results_datas.append(x_data + wgn(x_data, noise_power))

    return np.array(results_datas)

def wgns(x_datas, snr, std = 1):
    '''高斯白噪声
    '''
    def wgn(x, snr):
        snr = snr + (np.random.rand()-1)*std
        snr = 10**(snr/10.0)
        xpower = np.sum(x**2)/len(x)
        npower = xpower / snr
        return np.random.randn(len(x)) * np.sqrt(npower)

    results_datas = []
    for x_data in x_datas:
        results_datas.append(x_data + wgn(x_data, snr))

    return np.array(results_datas)


def pepperand_salt_noisy(x_datas, noise_probability):
    '''椒盐噪声
    '''
    noisy = np.random.binomial(1, 1-noise_probability, x_datas.shape)
    noisy = x_datas*noisy
    return noisy


def evaluate_wgn(signal, noise_signal):
    '''评估信号噪声功率
    '''
    xpower = np.sum(signal**2)/len(signal)

    ypower = np.sum((noise_signal-signal)**2)/len(noise_signal)
    return 10*math.log10(xpower/ypower)


def evaluate_euclidean_distance(signal, noise_signal):
    '''评估欧氏距离
    '''
    return np.sqrt(np.sum((signal-noise_signal)**2)/len(signal))

def test_acc(model, dataloader, device, total):
    num_correct = 0
    model.eval()
    for i, data in enumerate(dataloader, 0):
        test_signal, test_label = data
        test_signal, test_label = test_signal.to(device), test_label.to(device)
        pred_lab = torch.argmax(model(test_signal), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)
    return num_correct/total
    

def confusion_matrix(conf_arr, label_num):
    N = label_num

    # conf_arr = [[305., 0., 0., 0.],
    #             [0., 304., 0., 2.],
    #             [0., 0., 302., 0.],
    #             [0., 0., 3., 304.]]

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap="Blues",
                    interpolation='nearest')

    for i in range(0, N):
        for j in range(0, N):
            plt.annotate(round(np.array(norm_conf)[i, j], 4), xy=(
                i, j), horizontalalignment='center',   verticalalignment='center')

    width = len(conf_arr)
    height = len(conf_arr[0])
    cb = fig.colorbar(res)
    alphabet = ['N', 'B', 'IR', 'OR']
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    locs, labels = plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.tight_layout()
    plt.show()
    # plt.savefig('test.pdf')