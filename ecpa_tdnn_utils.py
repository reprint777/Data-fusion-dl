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

def train_loop(dataloader, model, device, loss, optimizer, loss_sum, accuracies):
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_id, (train_x, labels) in loop:
        # train_x = Variable(train_data[:,:,:128], requires_grad=True)
        # labels = train_data[:, :, 128][:, 0]
        spec_mag = train_x.to(device, dtype=torch.float)
        label = labels.to(device).long()
        # print(type(label), label.shape)
        output = model(spec_mag)
        # 计算损失值
        # print(type(output), output.shape)
        los = loss(output, label)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        loss_sum.append(los)
        # 计算准确率
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        print(type(output), output.shape, type(label), label.shape)
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc)
        loss_sum.append(los)

        loop.set_description()
        loop.set_postfix()

        if batch_id == len(dataloader) - 1:
            print(f'[{datetime.datetime.now()}],'
                  f'loss: {sum(loss_sum) / len(loss_sum):.8f},'
                  f'accuracy: {sum(accuracies) / len(accuracies):.8f}')

def test_loop(dataloader, model, device, loss, loss_sum, accuracies):
    model.eval()
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_id, (test_data, labels_y) in loop:
        test_x = test_data.to(device, dtype=torch.float)
        labels_test = labels_y.to(device).long()
        output = model(test_x)
        los = loss(output, labels_test)
        # 计算准确率
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = labels_test.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc)
        loss_sum.append(los)

        loop.set_description()
        loop.set_postfix()

        if batch_id == len(dataloader) - 1:
            print(f'[{datetime.datetime.now()}],'
                  f'loss: {sum(loss_sum) / len(loss_sum):.8f},'
                  f'accuracy: {sum(accuracies) / len(accuracies):.8f}')