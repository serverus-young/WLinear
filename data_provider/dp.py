import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.fft import fft, ifft, fftshift, ifftshift
import torch.nn as nn
import torch

class Dataset_ETTH(Dataset):
    def __init__(self,args,flag):
        self.args = args
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pre_len = args.pred_len
        self.data_path = args.data_path
        self.freq ='h'
        self.flag = flag
        assert self.flag in ["train", "val", "test"]
        type = {'train': 0, 'val': 1, 'test': 2}
        self.type = type[self.flag]
        self.features = args.features
        self.target = 'OT'
        self.scale = StandardScaler()
        self.__read__data__()

    def __read__data__(self):
        df_raw = pd.read_csv(os.path.join(self.args.data_path,))

        if self.args.data == 'ETTh':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.args.data == 'ETTm':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.args.data == 'custom':
            cols = list(df_raw.columns)
            cols.remove('OT')
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + ['OT']]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.type]
        border2 = border2s[self.type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        train_data = df_data[border1s[0]:border2s[0]]
        self.scale.fit(train_data.values)
        data = self.scale.transform(df_data.values)
        # 数据可视化
        # 1、etth1数据集去除趋势后，acf体现出明显的24周期的特性
        # 2、acf是经过归一化的，所以fft需要除以[0]才可以得到相同的图形
        # 3、
        # data = data[100:1000,0]
        # data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        # kernel_size = 97
        # trend = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)(data_tensor)
        # trend1 = trend.squeeze().numpy()  # 转回 numpy 数组
        # data1 = data - trend1
        # #
        # # # ReVIN
        # a = acf(data1,nlags=192)
        # plt.figure(figsize=(20,21))
        # plt.subplot(3,1,1)
        # plt.plot(a,label='acf')
        # plt.grid()
        # plt.subplot(3,1,2)
        # plt.plot(data1[:192],label='data')
        # plt.grid()
        # plt.subplot(3, 1, 3)
        # plt.plot(trend1[:192],label='data')
        # plt.grid()
        # plt.show()
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        # 提取年份、月份、星期几、日、小时列
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pre_len + 1
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pre_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # return的就是dataloder的输出
        return seq_x, seq_x_mark,seq_y, seq_y_mark
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def create_dataset_and_dataloader(args, flag, shuffle):
    dataset = Dataset_ETTH(args,flag)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=args.num_workers,drop_last=True)
    return dataset, dataloader



