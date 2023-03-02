from torch import nn
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from tqdm import tqdm
import copy
from torch.autograd import Variable
import librosa
import librosa.display
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
import os
import datetime
from torch.utils.data import DataLoader, SubsetRandomSampler

from ecapa_tdnn import EcapaTdnn
from dataaugmentation import dataaugment
from Sqential_utils import train_loop, test_loop, extract_tensor
from graphic_made import ConfusionMatrix
from Pre_imu import Pre_imu
from Pre_acoustic import Pre_acoustic
import random
import warnings

'''
class baseLSTMmodel(nn.Module):
    def __init__(self, inputSize, hiddenSize, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.LSTM(inputSize, hiddenSize, 1, batch_first=True),
            extract_tensor(),
            nn.Linear(hiddenSize, output_size))
'''


class Concantenation(nn.Module):
    def __init__(self, model1, model2):
        super(Concantenation, self).__init__()
        self.imupart = model1
        self.voicepart = model2

    def forward(self, x, y):
        out1 = self.imupart(x)
        out2 = self.voicepart(y)

        return (out1 + out2) / 2.0


class Concantenation2(nn.Module):
    def __init__(self, inputsize1, hiddensize1, inputsize2, hiddensize2, outsize):
        super(Concantenation2, self).__init__()
        self.lstm1 = nn.LSTM(input_size=inputsize1, hidden_size=hiddensize1)
        self.lstm2 = nn.LSTM(input_size=inputsize2, hidden_size=hiddensize2)
        self.linear = nn.Linear(in_features=(hiddensize1 + hiddensize2), out_features=outsize)

    def forward(self, x, y):
        out1,(h1, c1) = self.lstm1(x)
        out2,(h2, c2) = self.lstm2(y)
        # print(out1.shape, out2.shape)
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]
        z = torch.cat((out1, out2), 1)
        out = self.linear(z)

        return out


def averagetrain(dataloader, model, device, loss, optimizer, loss_sum, accuracies, hn, cn):
    model.train()
    for batch_id, (train_x, train_y, labels_x) in enumerate(dataloader):
        train_imu = train_x.to(device, dtype=torch.float)
        train_voi = train_y.to(device, dtype=torch.float)
        # print(type(spec_mag), spec_mag.shape, hn.shape, cn.shape)
        label = labels_x.to(device).long()
        # print(type(label), label.shape, hn.shape, cn.shape)
        output = model(train_imu, train_voi)
        # 计算损失值
        # print(type(output), output.shape, hn.shape, cn.shape)
        los = loss(output, label)
        hn = hn.detach()
        cn = cn.detach()
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        loss_sum.append(los)
        # 计算准确率
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        # print(type(output), output.shape, type(label), label.shape)
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc)
        loss_sum.append(los)

        if batch_id == len(dataloader) - 1:
            print(f'[{datetime.datetime.now()}],'
                  f'loss: {sum(loss_sum) / len(loss_sum):.8f},'
                  f'accuracy: {sum(accuracies) / len(accuracies):.8f}')


def averagetest(dataloader, model, device, loss, loss_sum, accuracies, hn, cn):
    for (batch_id, (test_x, test_y, labels_x)) in enumerate(dataloader):
        test_imu = test_x.to(device, dtype=torch.float)
        test_voi = test_y.to(device, dtype=torch.float)
        labels_test = labels_x.to(device).long()
        output = model(test_imu, test_voi)
        los = loss(output, labels_test)
        # 计算准确率

        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = labels_test.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc)
        loss_sum.append(los)

        if batch_id == len(dataloader) - 1:
            print(f'[{datetime.datetime.now()}],'
                  f'loss: {sum(loss_sum) / len(loss_sum):.8f},'
                  f'accuracy: {sum(accuracies) / len(accuracies):.8f}')
