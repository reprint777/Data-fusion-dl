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
from Concatenation import Concantenation, averagetest, averagetrain, Concantenation2
import random
import warnings

warnings.filterwarnings("ignore")

res = {'di': 0, 'ft': 1, 'hw': 2}
imu_sample_rate = 100
ac_sample_rate = 22050
frame_time = 2
imu_sample = frame_time * imu_sample_rate
ac_sample = frame_time * ac_sample_rate

# data loading
imu_dataset, imu_label = Pre_imu(imu_sample)
# acoustic_dataset, acoustic_label = Pre_acoustic(ac_sample)
acoustic_data = np.load("voicedata2s.npy")
acoustic_label = np.load("labeldata2s.npy")
print(pd.DataFrame(imu_label).value_counts())
acoustic_dataset = []
print(acoustic_data.shape, acoustic_label.shape)


# acoustic features extract
def features_extractor(audio):
    melspec = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    return logmelspec.T


def logmelpic(audio):
    melspec = librosa.feature.melspectrogram(y=audio, sr=22050)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

for i in acoustic_data:
    acoustic_dataset.append(features_extractor(i))

# "other"delete
a = []
for i in range(0, len(imu_label)):
    if imu_label[i] != 3:
        a.append(i)
print(a)
imu_dataset3, imu_label3, acoustic_dataset3 = [], [], []
p = 0
for b in a:
    imu_dataset3.append(imu_dataset[b])
    imu_label3.append(imu_label[b])
    acoustic_dataset3.append(acoustic_dataset[b])
    if p < 20:
        logmelpic(acoustic_data[b])
        p += 1

# dataaug

imu_datasetfi, imu_labelfi, acoustic_datasetfi = [], [], []
for k in range(0, len(imu_dataset3)):
    A = dataaugment(imu_dataset3[k])
    B = dataaugment(acoustic_dataset3[k])
    RJ = random.uniform(0.05, 0.1)
    RS = random.uniform(0.1, 0.3)
    RT = random.uniform(0.1, 0.4)
    if imu_label3[k] == 0:
        t = 40
    elif imu_label3[k] == 2:
        t = 20
    else:
        t = 10
    for j in range(t):
        imu_datasetfi.append(A.DA_Jitter(RJ).data)
        acoustic_datasetfi.append(B.DA_Jitter(RJ * 2).data)
        imu_labelfi.append(imu_label3[k])

        imu_datasetfi.append(A.DA_Scaling(RS).data)
        acoustic_datasetfi.append(B.DA_Scaling(RS * 2).data)
        imu_labelfi.append(imu_label3[k])

        imu_datasetfi.append(A.DA_TimeWarp(RT, 4).data)
        acoustic_datasetfi.append(B.DA_TimeWarp(RT * 2, 4).data)
        imu_labelfi.append(imu_label3[k])

    imu_datasetfi.append(imu_dataset3[k])
    acoustic_datasetfi.append(acoustic_dataset3[k])
    imu_labelfi.append(imu_label3[k])

print(torch.tensor(acoustic_dataset).shape)
print(pd.DataFrame(imu_labelfi).value_counts())

Dataset1 = Data.TensorDataset(torch.tensor(imu_datasetfi), torch.tensor(acoustic_datasetfi), torch.tensor(imu_labelfi))
# Dataset2 = Data.TensorDataset(torch.tensor(acoustic_datasetfi), torch.tensor(imu_labelfi))
# print(imu_dataset.shape, acoustic_dataset.shape)
# divide dataset

shuffled_indices = np.random.permutation(len(Dataset1))
train_idx = shuffled_indices[:int(0.8 * len(Dataset1))]
test_idx = shuffled_indices[int(0.8 * len(Dataset1)):]

batch_size = 128
train_loader = DataLoader(Dataset1, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx))
test_loader = DataLoader(Dataset1, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(test_idx))

NUM_CLASSES = 3
Input_size = 6
Hidden_size = 8
Num_layers = 1

Input_size2 = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model1 = nn.Sequential(
    nn.LSTM(Input_size, Hidden_size, 1, batch_first=True),
    extract_tensor(),
    nn.Linear(Hidden_size, NUM_CLASSES))

model2 = nn.Sequential(
    nn.LSTM(Input_size2, Hidden_size, 1, batch_first=True),
    extract_tensor(),
    nn.Linear(Hidden_size, NUM_CLASSES))

model3 = Concantenation(model1=model1, model2=model2)

model = Concantenation2(inputsize1=6, hiddensize1=8, inputsize2=128, hiddensize2=8, outsize=NUM_CLASSES)

optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001,
                             weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
loss = torch.nn.CrossEntropyLoss()

epoch_number = 40
for epoch in range(epoch_number):
    print(f'Train epoch [{epoch}/{epoch_number}]')
    hn = torch.zeros(Num_layers, batch_size, Hidden_size)
    cn = torch.zeros(Num_layers, batch_size, Hidden_size)
    averagetrain(train_loader, model, device, loss, optimizer, [], [], hn, cn)
    print(f'lr: {scheduler.get_last_lr()[0]:.8f}')
    averagetest(test_loader, model, device, loss, [], [], hn, cn)
    scheduler.step()

class_indict = res
label = [_ for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=label)
# 实例化混淆矩阵，这里NUM_CLASSES = 8
with torch.no_grad():
    model.eval()  # 验证
    hn = torch.zeros(Num_layers, batch_size, Hidden_size)
    cn = torch.zeros(Num_layers, batch_size, Hidden_size)
    for j, (inputs, inputs2, labels) in enumerate(test_loader):
        inputs = inputs.to(device, dtype=torch.float)
        inputs2 = inputs2.to(device, dtype=torch.float)
        labels = labels.to(device).long()
        output = model(inputs, inputs2)  # 分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
        los = loss(output, labels)
        # valid_loss += los.item() * inputs.size(0)
        ret, predictions = torch.max(output.data, 1)
        # torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
        # confusion_matrix
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
    confusion.plot()
    confusion.summary()
