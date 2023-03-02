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
from LSTM_utils import test_loop, train_loop
from graphic_made import ConfusionMatrix
import random

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
# print(labelarray)
# print(timearray)

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
    acoustic_features.replace(['cw', 'dw', 'mw', 'mo'], 'other', inplace=True)
    acoustic_features = acoustic_features.drop(acoustic_features[(acoustic_features[128] == 'other')].index)
    label = LabelEncoder()
    acoustic_features[128] = label.fit_transform(acoustic_features[128])
    dfs.append(acoustic_features)

res = {}
for cl in label.classes_:
    res.update({cl: label.transform([cl])[0]})
print(res, len(res))

WINDOW_SIZE = 100
SLIDE_SIZE = 50
voice_dataset = None
label_dataset = None


# slide_function
def dataaugjitter(data):
    D = dataaugment(data.values.astype(float))
    sigma = random.uniform(0.05, 0.1)
    A = pd.DataFrame((D.DA_Jitter(sigma)).data)
    # print(A.shape)
    return A


def dataaugscaling(data):
    D = dataaugment(data.values.astype(float))
    sigma = random.uniform(0.1, 0.3)
    A = pd.DataFrame((D.DA_Scaling(sigma)).data)
    # print(A.shape)
    return A


def dataaugtimewrap(data):
    D = dataaugment(data.values.astype(float))
    sigma = random.uniform(0.1, 0.4)
    A = pd.DataFrame((D.DA_TimeWarp(sigma, 4)).data)
    # print(A.shape)
    return A


def get_frame(dataframe, window_size, slide_size):
    length = len(dataframe)
    voice_datas = []
    label_datas = []
    for i in range(int((length - window_size) / slide_size)):
        voice_data = dataframe.iloc[i * slide_size: i * slide_size + window_size, 0:128]
        label_data = stats.mode(dataframe[128][i * slide_size: i * slide_size + window_size])[0][0]

        ran = random.random()
        if label_data == res['hw']:
            t = 23
        elif label_data == res['di']:
            t = 40
        else:
            t = 10
        #    continue
        #elif label_data == res['ft'] and ran > 0.33:
        #    continue

        # if label_data:
        for a in range(t):
            voice_datas.append(dataaugjitter(voice_data))
            label_datas.append(label_data)
        for b in range(t):
            voice_datas.append(dataaugscaling(voice_data))
            label_datas.append(label_data)
        for c in range(t):
            voice_datas.append(dataaugtimewrap(voice_data))
            label_datas.append(label_data)
         #   continue

        voice_datas.append(voice_data)
        label_datas.append(label_data)
    return voice_datas, label_datas


for i in dfs:
    x, y = get_frame(i, WINDOW_SIZE, SLIDE_SIZE)
    if voice_dataset is None:
        voice_dataset = x
        label_dataset = y
        # print(voice_dataset[0].shape, type(voice_dataset[0]), type(label_dataset[0]))
    else:
        voice_dataset = np.concatenate((voice_dataset, x), axis=0)
        label_dataset = np.concatenate((label_dataset, y), axis=0)

# torch.tensor data generating
voice_dataset = voice_dataset.astype(float)
label_dataset = label_dataset.astype(float)

print(pd.DataFrame(label_dataset).value_counts())
Dataset = Data.TensorDataset(torch.tensor(voice_dataset), torch.tensor(label_dataset))

# divide dataset
shuffled_indices = np.random.permutation(len(Dataset))
train_idx = shuffled_indices[:int(0.8 * len(Dataset))]
test_idx = shuffled_indices[int(0.8 * len(Dataset)):]

batch_size = 128
train_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx))
test_loader = DataLoader(Dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(test_idx))

# modeling

NUM_CLASSES = len(res)
Hidden_size = 3
Num_layers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.LSTM(input_size=128, hidden_size=Hidden_size, num_layers=Num_layers, batch_first=True)
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001,
                             weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss = torch.nn.CrossEntropyLoss()
epoch_number = 20
for epoch in range(epoch_number):
    print(f'Train epoch [{epoch}/{epoch_number}]')
    hn = torch.zeros(Num_layers, batch_size, Hidden_size)
    cn = torch.zeros(Num_layers, batch_size, Hidden_size)
    train_loop(train_loader, model, device, loss, optimizer, [], [], hn, cn)
    print(f'lr: {scheduler.get_last_lr()[0]:.8f}')
    test_loop(test_loader, model, device, loss, [], [], hn, cn)
    scheduler.step()

class_indict = res
label = [_ for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=label)
# 实例化混淆矩阵，这里NUM_CLASSES = 8
with torch.no_grad():
    model.eval()  # 验证
    hn = torch.zeros(Num_layers, batch_size, Hidden_size)
    cn = torch.zeros(Num_layers, batch_size, Hidden_size)
    for j, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device).long()
        output, (hn, cn) = model(inputs, (hn, cn))  # 分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
        output = output[:, 0, :]
        los = loss(output, labels)
        # valid_loss += los.item() * inputs.size(0)
        ret, predictions = torch.max(output.data, 1)
        # torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
        # confusion_matrix
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
    confusion.plot()
    confusion.summary()