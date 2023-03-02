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
from ecpa_tdnn_utils import test_loop, train_loop

path = "/Users/haoyu/Desktop/fish2/HCII2023/acoustic"
pathlabel = "/Users/haoyu/Desktop/fish2/HCII2023/labelhelp"
filenameList = []
labelfile = []

for root, dirs, files in os.walk(path):
    for file in files:
        if '.wav' in file:
            filenameList.append(file)
# print(filenameList)

for root, dirs, files in os.walk(pathlabel):
    for file in files:
        if '.csv' in file:
            labelfile.append(file)
# print(labelfile)


labelarray = []
timearray = []
for j in labelfile:
    df = pd.read_csv(pathlabel + '/' + j)
    for k in range(0, df.shape[0]):
        labelarray.append(np.array(df.iloc[k, 0]).tolist())
        timearray.append(np.array(df.iloc[k, 1:3]).tolist())
print(labelarray)
print(timearray)

starttimestamp = {'feng': 1674017667.00, 'tang': 1674026819.82, 'zhuang': 1673940351.57, 'han': 1674005330.12}


def features_extractor(file, str):
    audio, sample_rate = librosa.load(file, sr=22050)
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)

    x = [0] * len(logmelspec[0])
    y = ['other'] * len(logmelspec[0])
    for i in range(0, len(logmelspec[0])):
        x[i] = (1024.0 + 512.0 * i) / 22050.0 + starttimestamp[str]
        for p in range(0, len(timearray)):
            if timearray[p][0] <= x[i] <= timearray[p][1]:
                y[i] = labelarray[p]
    logmelspec = np.vstack([logmelspec, y])
    return logmelspec.T


dfs = []
for i in filenameList:
    f = features_extractor(path + '/' + i, i.split('.')[0])
    acoustic_features = pd.DataFrame(f)
    label = LabelEncoder()
    acoustic_features[128] = label.fit_transform(acoustic_features[128])
    dfs.append(acoustic_features)

WINDOW_SIZE = 100
SLIDE_SIZE = 50
voice_dataset = None
label_dataset = None


# slide_function
def dataaugjitter(data):
    D = dataaugment(data.values.astype(float))
    A = pd.DataFrame((D.DA_Jitter(0.05)).data)
    # print(A.shape)
    return A


def dataaugscaling(data):
    D = dataaugment(data.values.astype(float))
    A = pd.DataFrame((D.DA_Scaling(0.1)).data)
    # print(A.shape)
    return A


def dataaugtimewrap(data):
    D = dataaugment(data.values.astype(float))
    A = pd.DataFrame((D.DA_TimeWarp(0.2, 4)).data)
    # print(A.shape)
    return A


def get_frame(dataframe, window_size, slide_size):
    length = len(dataframe)
    voice_datas = []
    label_datas = []
    for i in range(int((length - window_size) / slide_size)):
        voice_data = dataframe.iloc[i * slide_size: i * slide_size + window_size, 0:128]
        label_data = stats.mode(dataframe[128][i * slide_size: i * slide_size + window_size])[0][0]

        if label_data != 7:
            for a in range(8):
                voice_datas.append(dataaugjitter(voice_data))
                label_datas.append(label_data)
            for b in range(8):
                voice_datas.append(dataaugscaling(voice_data))
                label_datas.append(label_data)
            for c in range(8):
                voice_datas.append(dataaugtimewrap(voice_data))
                label_datas.append(label_data)
            continue

        voice_datas.append(voice_data)
        label_datas.append(label_data)
    return voice_datas, label_datas


for i in dfs:
    x, y = get_frame(i, WINDOW_SIZE, SLIDE_SIZE)
    if voice_dataset is None:
        voice_dataset = x
        label_dataset = y
        print(voice_dataset[0].shape, type(voice_dataset[0]), type(label_dataset[0]))
    else:
        print(type(label_dataset), type(y))
        voice_dataset = np.concatenate((voice_dataset, x), axis=0)
        label_dataset = np.concatenate((label_dataset, y), axis=0)

# torch.tensor data generating
voice_dataset = voice_dataset.astype(float)
label_dataset = label_dataset.astype(float)
Dataset = Data.TensorDataset(torch.tensor(voice_dataset), torch.tensor(label_dataset))

# divide dataset
shuffled_indices = np.random.permutation(len(Dataset))
train_idx = shuffled_indices[:int(0.8 * len(Dataset))]
test_idx = shuffled_indices[int(0.8 * len(Dataset)):]

batch_size = 128
train_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx))
test_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(test_idx))

# modeling

device = torch.device("cpu")
model = EcapaTdnn(num_classes=8, input_size=128).to(device)
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
