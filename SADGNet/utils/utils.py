import torch
from torch import nn
import numpy as np
from SADGNet.utils.logger import my_logger
from copy import deepcopy
import math


def act_str2obj(activation):
    act = nn.ReLU()
    if activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Sigmoid':
        act = nn.Sigmoid()
    elif activation == 'Tanh':
        act = nn.Tanh()
    return act


def get_optimizer(model, lr, optimizer_str):
    if optimizer_str == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_str == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_str == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer ' + model.optimizer +
                                  ' is not implemented')


class DiscreteLabels1:
    def __init__(self):
        self.all_time_index = None
        self.time_index = None
        self.time_index_id = None
        self.max_e = None
        self.max_t = None
        self.mask1 = None
        self.mask2 = None

    def fit(self, train_label, time_interval=6, time_index=None):
        self.max_e = int(np.max(train_label[:, 1]))
        self.max_t = math.ceil(np.max(train_label[:, 0]))
        if time_index is None:
            self.time_index = np.arange(time_interval, self.max_t, time_interval)
        else:
            self.time_index = time_index
        self.time_index_id = list(range(len(self.time_index)))

    def transform(self, test_label):
        sample_num = test_label.shape[0]
        times = test_label[:, 0]
        events = test_label[:, 1]
        ret = np.ones((sample_num, len(self.time_index)))
        for i in range(sample_num):
            for j in range(len(self.time_index)):
                if j == len(self.time_index) - 1:
                    j_add_1 = self.max_t+10000
                else:
                    j_add_1 = self.time_index[j + 1]
                if events[i] == 1:
                    if times[i] >= j_add_1:
                        ret[i, j] = 0
                    elif times[i] >= self.time_index[j]:
                        ret[i, j] = 1
                    elif times[i] < self.time_index[j]:
                        ret[i, j] = 1
                elif events[i] == 0:
                    if times[i] >= j_add_1:
                        ret[i, j] = 0
                    elif times[i] >= self.time_index[j]:
                        ret[i, j] = 0
                    elif times[i] < self.time_index[j]:
                        ret[i, j] = self.max_e + 1
        mask1 = np.zeros((sample_num, len(self.time_index)))
        mask2 = np.zeros((sample_num, len(self.time_index)))
        for i in range(ret.shape[0]):
            first_flag = True
            for j in range(ret.shape[1]):
                if events[i] == 1:
                    if ret[i, j] == 0:
                        mask1[i, j] = 0
                        mask2[i, j] = 1
                    elif ret[i, j] == 1:
                        if first_flag:
                            mask1[i, j] = 1
                            mask2[i, j] = 1
                            first_flag = False
                        else:
                            mask1[i, j] = 0
                            mask2[i, j] = 0
                elif events[i] == 0:
                    if ret[i, j] == 0:
                        mask1[i, j] = 1
                        mask2[i, j] = 1
                    elif ret[i, j] == 2:
                        mask1[i, j] = 0
                        mask2[i, j] = 0
        self.mask1 = mask1
        self.mask2 = mask2
        return ret


class DiscreteLabels2:
    def __init__(self):
        self.all_time_index = None
        self.time_index = None
        self.time_index_id = None
        self.max_e = None
        self.max_t = None
        self.mask1 = None
        self.mask2 = None

    def fit(self, train_label, time_interval=6, time_index=None):
        self.max_e = int(np.max(train_label[:, 1]))
        self.max_t = math.ceil(np.max(train_label[:, 0]))
        if time_index is None:
            self.time_index = np.arange(time_interval, self.max_t, time_interval)
        else:
            self.time_index = time_index
        self.time_index_id = list(range(len(self.time_index)))

    def get_mask(self, test_label):
        sample_num = test_label.shape[0]
        times = test_label[:, 0]
        events = test_label[:, 1]
        ret = np.ones((sample_num, len(self.time_index)))
        for i in range(sample_num):
            for j in range(len(self.time_index)):
                if j == len(self.time_index) - 1:
                    j_add_1 = self.max_t+10000
                else:
                    j_add_1 = self.time_index[j + 1]
                if events[i] == 1:
                    if times[i] >= j_add_1:
                        ret[i, j] = 0
                    elif times[i] >= self.time_index[j]:
                        ret[i, j] = 1
                    elif times[i] < self.time_index[j]:
                        ret[i, j] = 1
                elif events[i] == 0:
                    if times[i] >= j_add_1:
                        ret[i, j] = 0
                    elif times[i] >= self.time_index[j]:
                        ret[i, j] = 0
                    elif times[i] < self.time_index[j]:
                        ret[i, j] = self.max_e + 1
        mask1 = np.zeros((sample_num, len(self.time_index)))
        mask2 = np.zeros((sample_num, len(self.time_index)))
        for i in range(ret.shape[0]):
            first_flag = True
            for j in range(ret.shape[1]):
                if events[i] == 1:
                    if ret[i, j] == 0:
                        mask1[i, j] = 0
                        mask2[i, j] = 1
                    elif ret[i, j] == 1:
                        if first_flag:
                            mask1[i, j] = 1
                            mask2[i, j] = 1
                            first_flag = False
                        else:
                            mask1[i, j] = 0
                            mask2[i, j] = 0
                elif events[i] == 0:
                    if ret[i, j] == 0:
                        mask1[i, j] = 1
                        mask2[i, j] = 1
                    elif ret[i, j] == 2:
                        mask1[i, j] = 0
                        mask2[i, j] = 0
        return mask1, mask2

    def transform(self, label):
        label1 = label.copy()
        label2 = label.copy()
        label1[:, 1] = (label[:, 1] == 1).astype(int)
        label2[:, 1] = (label[:, 1] == 2).astype(int)
        mask1_1, mask1_2 = self.get_mask(label1)
        mask2_1, mask2_2 = self.get_mask(label2)
        self.mask1 = np.stack((mask1_1, mask2_1), axis=1)
        self.mask2 = np.stack((mask1_2, mask2_2), axis=1)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, dics):
        score = -val_loss
        if np.isnan(val_loss):
            self.counter += 1
            my_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.best_score is None:
            self.best_score = score
            dics[0] = deepcopy(model.state_dict())
        elif score <= self.best_score + self.delta:
            self.counter += 1
            my_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            dics[0] = deepcopy(model.state_dict())


