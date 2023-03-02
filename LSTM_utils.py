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
from ecapa_tdnn import EcapaTdnn
import datetime
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataaugmentation import dataaugment


def train_loop(dataloader, model, device, loss, optimizer, loss_sum, accuracies, hn, cn):
    size = len(dataloader.dataset)
    model.train()
    for batch_id, (train_x, labels) in enumerate(dataloader):
        spec_mag = train_x.to(device, dtype=torch.float)
        # print(type(spec_mag), spec_mag.shape, hn.shape, cn.shape)
        label = labels.to(device).long()
        # print(type(label), label.shape, hn.shape, cn.shape)
        output, (hn, cn) = model(spec_mag, (hn, cn))
        # 计算损失值
        output = output[:, 0, :]
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


def test_loop(dataloader, model, device, loss, loss_sum, accuracies, hn, cn):
    for batch_id, (test_data, labels_y) in enumerate(dataloader):
        test_x = test_data.to(device, dtype=torch.float)
        labels_test = labels_y.to(device).long()
        output, (hn, cn) = model(test_x, (hn, cn))
        output = output[:, 0, :]
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
