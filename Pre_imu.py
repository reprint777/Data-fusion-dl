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


def Pre_imu(imu_sample):
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
    filenameList.sort()
    print(filenameList)
    dfs = []
    for i in filenameList:
        df = pd.read_csv(path + '/' + i, dtype={'label': 'str'})  # read file
        df.drop(['timestamp', 'hr', 'barometer', 'gravity-x', 'gravity-y',
                 'gravity-z', 'rotation-x', 'rotation-y', 'rotation-z', 'user-acc-x', 'user-acc-y', 'user-acc-z'],
                axis=1, inplace=True)  # delete arrays that is not needed
        df.dropna(axis=0, how='all', inplace=True)  # delete nan
        df.replace(['cw', 'dw', 'mw', 'mo'], 'other', inplace=True)
        # df = df.drop(df[df['label'] == 'other'].index)
        label = LabelEncoder()
        df['label'] = label.fit_transform(df['label'])  # transform label from str to num
        dfs.append(df)
        print(i)

    res = {}
    for cl in label.classes_:
        res.update({cl: label.transform([cl])[0]})
    print(res, len(res))

    WINDOW_SIZE = imu_sample
    SLIDE_SIZE = imu_sample//2
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
            '''
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
            '''
            imu_datas.append(imu_data)
            label_datas.append(label_data)
        print(len(imu_datas))
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
    print(imu_dataset.shape, label_dataset.shape)
    np.save("imudata2s.npy", imu_dataset)
    np.save("imulabel2s.npy", label_dataset)
    return imu_dataset, label_dataset
