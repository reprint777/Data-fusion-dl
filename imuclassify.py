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
from ecpa_tdnn_utils import train_loop, test_loop
from graphic_made import ConfusionMatrix
import random


print(torch.cuda.is_available(), torch.__version__, torch.version.cuda, torch.cuda.device_count())

path = "/Users/haoyu/Desktop/fish2/HCII2023/imudata"
Behaviors = {"other": 0, "cw": 1, "di": 2, "dw": 3, "hw": 4, "mw": 5, "mo": 6, "ft": 7}
IMU_columns = ['label', 'accx', 'accy', 'accz', 'roll', 'pitch', 'yaw', 'gravity-x', 'gravity-y',
               'gravity-z', 'rotation-x', 'rotation-y', 'rotation-z', 'user-acc-x', 'user-acc-y', 'user-acc-z']
filenameList = []
data_imu = pd.DataFrame(columns=IMU_columns)
for root, dirs, files in os.walk(path):
    for file in files:
        if '.csv' in file:
            filenameList.append(file)

dfs = []
for i in filenameList:
    df = pd.read_csv(path + '/' + i, dtype={'label': 'str'})  # read file
    df.drop(['timestamp', 'hr', 'barometer', 'gravity-x', 'gravity-y',
             'gravity-z', 'rotation-x', 'rotation-y', 'rotation-z', 'user-acc-x', 'user-acc-y', 'user-acc-z'],
            axis=1, inplace=True)  # delete arrays that is not needed
    df.dropna(axis=0, how='all', inplace=True)  # delete nan
    df.replace(['cw', 'dw', 'mw', 'mo'], 'other', inplace=True)

    label = LabelEncoder()
    df['label'] = label.fit_transform(df['label'])  # transform label from str to num
    dfs.append(df)

res = {}
for cl in label.classes_:
    res.update({cl: label.transform([cl])[0]})
print(res, res['other'], len(res))

WINDOW_SIZE = 100
SLIDE_SIZE = 50
imu_dataset = None
label_dataset = None


# slide_function and data augmentation
def dataaugjitter(data):
    D = dataaugment(data.values)
    sigma = random.uniform(0.05, 0.1)
    A = pd.DataFrame((D.DA_Jitter(sigma)).data)
    # print(A.shape)
    return A


def dataaugscaling(data):
    D = dataaugment(data.values)
    sigma = random.uniform(0.1, 0.3)
    A = pd.DataFrame((D.DA_Scaling(sigma)).data)
    # print(A.shape)
    return A


def dataaugtimewrap(data):
    D = dataaugment(data.values)
    sigma = random.uniform(0.1, 0.4)
    A = pd.DataFrame((D.DA_TimeWarp(sigma, 4)).data)
    # print(A.shape)
    return A


def get_frame(dataframe, window_size, slide_size):
    length = len(dataframe)
    imu_datas = []
    label_datas = []
    for i in range(int((length - window_size) / slide_size)):
        imu_data = dataframe.iloc[i * slide_size: i * slide_size + window_size, 1:]
        label_data = stats.mode(dataframe['label'][i * slide_size: i * slide_size + window_size])[0][0]
        ran = random.random()
        if label_data == res['other'] and ran > 0.25:
            continue
        elif label_data == res['ft'] and ran > 0.33:
            continue
        if label_data != res['other']:
            for a in range(10):
                imu_datas.append(dataaugjitter(imu_data))
                label_datas.append(label_data)
            for b in range(10):
                imu_datas.append(dataaugscaling(imu_data))
                label_datas.append(label_data)
            for c in range(10):
                imu_datas.append(dataaugtimewrap(imu_data))
                label_datas.append(label_data)
            continue
        imu_datas.append(imu_data)
        label_datas.append(label_data)
    return imu_datas, label_datas


# numpy data generating
for i in dfs:
    x, y = get_frame(i, WINDOW_SIZE, SLIDE_SIZE)
    if imu_dataset is None:
        imu_dataset = x
        label_dataset = y
        # print(imu_dataset[0].shape)
    else:
        imu_dataset = np.concatenate((imu_dataset, x), axis=0)
        label_dataset = np.concatenate((label_dataset, y), axis=0)

# print(type(imu_dataset_train))
print(pd.DataFrame(label_dataset).value_counts())
# torch.tensor data generating
# print(imu_dataset, label_dataset, len(imu_dataset), len(label_dataset))
Dataset = Data.TensorDataset(torch.tensor(imu_dataset), torch.tensor(label_dataset))
print(len(Dataset), imu_dataset.shape)

# divide dataset
shuffled_indices = np.random.permutation(len(Dataset))
train_idx = shuffled_indices[:int(0.8 * len(Dataset))]
test_idx = shuffled_indices[int(0.8 * len(Dataset)):]

batch_size = 128
train_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx))
test_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(test_idx))

# modeling

NUM_CLASSES = len(res)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EcapaTdnn(num_classes=NUM_CLASSES, input_size=6).to(device)
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001,
                             weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss = torch.nn.CrossEntropyLoss()
epoch_number = 20
for epoch in range(epoch_number):
    print(f'Train epoch [{epoch}/{epoch_number}]')
    train_loop(train_loader, model, device, loss, optimizer, [], [])
    print(f'lr: {scheduler.get_last_lr()[0]:.8f}')
    test_loop(test_loader, model, device, loss, [], [])
    scheduler.step()

class_indict = res
label = [label for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=label)
# 实例化混淆矩阵，这里NUM_CLASSES = 8
with torch.no_grad():
    model.eval()  # 验证
    for j, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)  # 分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
        los = loss(output, labels)
        # valid_loss += los.item() * inputs.size(0)
        ret, predictions = torch.max(output.data, 1)
        # torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
        # confusion_matrix
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
    confusion.plot()
    confusion.summary()
