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


def Pre_acoustic(ac_sample):
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

    def data_cutting(file, str, size, slide):
        audios = []
        audio, sample_rate = librosa.load(file, sr=22050)
        timestamps, timestamp = [], [0] * len(audio)
        framelabels, framelabel = [], ['other'] * len(audio)
        for i in range(0, len(audio)):
            timestamp[i] = i / 22050.0 + starttimestamp[str]
            for p in range(0, len(timearray)):
                if timearray[p][0] <= timestamp[i] <= timearray[p][1]:
                    framelabel[i] = labelarray[p]
                    break
        dfaudio = pd.DataFrame(audio)
        dfaudio['label'] = framelabel
        dfaudio.replace(['cw', 'dw', 'mw', 'mo'], 'other', inplace=True)

        # dfaudio = dfaudio.drop(dfaudio[dfaudio['label'] == 'other'].index)
        label = LabelEncoder()
        dfaudio['label'] = label.fit_transform(dfaudio['label'])

        for j in range(0, int((len(audio) - size) / slide)):
            audios.append(audio[slide * j: slide * j + size])
            framelabels.append(stats.mode(dfaudio['label'].values[slide * j: slide * j + size])[0][0])
        return audios, framelabels


    WINDOW_SIZE = ac_sample
    SLIDE_SIZE = ac_sample // 2
    voice_dataset = None
    label_dataset = None

    dfs = []
    ls = []
    for i in filenameList:
        print(ac_sample, int(ac_sample / 2))
        audios, framelabels = data_cutting(path + '/' + i, i.split('.')[0], ac_sample, int(ac_sample / 2))
        dfs.append(audios)
        ls.append(framelabels)

    for i in range(0, len(dfs)):
        if voice_dataset is None:
            voice_dataset = dfs[i]
            label_dataset = ls[i]
            print(voice_dataset[0].shape, type(voice_dataset[0]), ls[i], dfs[i])
        else:
            voice_dataset = np.concatenate((voice_dataset, dfs[i]), axis=0)
            label_dataset = np.concatenate((label_dataset, ls[i]), axis=0)
    print(len(dfs), len(ls))

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



    # torch.tensor data generating
    voice_dataset = voice_dataset.astype(float)
    label_dataset = label_dataset.astype(float)
    print(voice_dataset.shape, label_dataset.shape)
    np.save("voicedata6s.npy", voice_dataset)
    np.save("labeldata6s.npy", label_dataset)
    return voice_dataset, label_dataset
