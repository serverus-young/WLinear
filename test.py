# -*- coding: utf-8 -*-
import os
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
import time
from tqdm import tqdm
from data_provider.dp import Dataset_ETTH
import torch
import torch.nn as nn
import numpy as np
from data_provider.dp import create_dataset_and_dataloader
from model.classic_model import Transformer
from model.PatchTST import PatchTST
from model.Patch2 import Patch2
from model.patch3 import Patch3
from model.itrans import iTrans
from model.TimesNet import TimesNet
from model.Sparse import Sparse
from model.WFT import WFT
from model.WT2 import WT2
from model.TimesNet2 import TimesNet2
from utils.tools import EarlyStopping , adjust_learning_rate, save_results
from utils.metric import metric
import random
from thop import profile
from model.FITS import FITS
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# fix_seed = 2024
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for training and testing")

    # 数据和任务相关参数
    parser.add_argument('--data_path', type=str, default='./dataset/ETT-small/ETTh1.csv', help='Path to the dataset')
    parser.add_argument('--data', type=str, default='ETTh', help='Dataset name')
    parser.add_argument('--Data', type=str, default='ETTh1', help='Dataset name')
    parser.add_argument('--seq_len', type=int, default=720, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction sequence length')
    parser.add_argument('--freq', type=str, default='h', help='Frequency for time-series data')
    parser.add_argument('--task', type=str, default='forecast', help='Task type (e.g., forecast, classification)')
    parser.add_argument('--features', type=str, default='MS',
                        help='Feature type (M for univariate, MS for multivariate)')

    # 模型相关参数
    parser.add_argument('--model', type=str, default='WFT', help='Model name')
    parser.add_argument('--d_model', type=int, default=16, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=32, help='Dimension of feed-forward network')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_type', type=str, default='conv', help='Feed-forward type (e.g., dense, conv)')
    parser.add_argument('--num_e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--e_in', type=int, default=7, help='Number of varible (ETT)')
    parser.add_argument('--c_in', type=int, default=7, help='Number of varible (ETT)')
    parser.add_argument('--num_d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for PatchTST or similar models')
    parser.add_argument('--k', type=int, default=5, help='Kernel size for TimesNet')
    parser.add_argument('--k2', type=int, default=5, help='Additional kernel size for TimesNet')
    parser.add_argument('--num_kernels', type=int, default=5, help='Number of kernels in TimesNet')
    parser.add_argument('--model_type', type=str, default='linear', help='Sparse model type (e.g., linear, nonlinear)')
    parser.add_argument('--scales', type=int, default=4, help='Number of scales for WFT')
    parser.add_argument('--num_scales', type=int, default=8, help='Number of scales for WFT')

    # 训练和优化参数
    parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--adjust_lr_type', type=str, default='exponential_decay', help='Learning rate adjustment type')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output during training')
    parser.add_argument('--save_results', type=bool, default=True, help='save_results')

    # 设备相关参数
    parser.add_argument('--device', type=int, default=0, help='GPU device index (0 for the first GPU)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='Path to save model checkpoints')

    # FITS
    parser.add_argument('--cut_freq', type=int, default=0, help='Path to save model checkpoints')
    parser.add_argument('--H_order', type=int, default=6, help='Path to save model checkpoints')
    parser.add_argument('--base_T', type=int, default=24, help='Path to save model checkpoints')
    # 解析参数
    args = parser.parse_args()
    return args

class Long_term_Forcast():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = self._build_model().to(args.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)


    def clean_state_dict(state_dict):
        # 过滤掉包含 "total_ops" 和 "total_params" 的键
        clean_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
        return clean_dict

    def measure_inference_time_on_gpu(self,model, input_tensors, repeat=100):
        model.eval()
        # 进行一次前向传播，确保模型在 GPU 上加载
        with torch.no_grad():
            model(*input_tensors)  # 使用 * 解包传递多个输入张量
        # 同步 GPU，确保所有操作已经完成
        torch.cuda.synchronize()
        # 开始计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(repeat):
                model(*input_tensors)  # 重复推理，传入多个输入张量
        # 同步 GPU，再次确保所有操作完成
        torch.cuda.synchronize()
        # 结束计时
        end_time = time.time()
        # 计算平均推理时间
        avg_inference_time = (end_time - start_time) / repeat
        print(f'Average Inference Time on GPU: {avg_inference_time:.6f} seconds')
        return avg_inference_time
    def _get_data(self,args,flag,shuffle):
        data_set, data_loader = create_dataset_and_dataloader(args,flag,shuffle)
        return data_set,data_loader
    def _build_model(self):
        model_dict = {
            'Transformer' : Transformer,
            'TimesNet' : TimesNet,
            'TimesNet2' : TimesNet2,
            'PatchTST' : PatchTST,
            'Patch2' :Patch2,
            'Patch3' :Patch3,
            'iTrans' :iTrans,
            'Sparse':Sparse,
            'WFT':WFT,
            'WT2':WT2,
            'FITS':FITS
        }
        model_class = model_dict.get(self.args.model,  Transformer)
        model = model_class(self.args).float()
        return model

    def test(self,args):
        test_data, test_loader = self._get_data(args,'test',shuffle=False)
        print('testing', len(test_data))
        print('loading model')
        filepath = os.path.join(args.checkpoint_path, self.args.model + '_' + self.args.data + '_' + str(self.args.seq_len) + '_' + str(
            self.args.pred_len) + '_' + 'ckpt_best.pth')
        self.model.load_state_dict(torch.load(filepath, weights_only=True),strict=False)
        self.model.eval()

        batch_x, batch_x_stamp, batch_y, batch_y_stamp = next(iter(test_loader))
        batch_x = batch_x.float().to(self.device)
        batch_x_stamp = batch_x_stamp.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_y_stamp = batch_y_stamp.to(self.device)
        inputs = (batch_x, batch_x_stamp, batch_y, batch_y_stamp)
        self.measure_inference_time_on_gpu(self.model, inputs)
        # 所有的数据累加起来一起算loss
        pred = []
        true = []
        with torch.no_grad():
            for i, (batch_x, batch_x_stamp, batch_y, batch_y_stamp) in enumerate(test_loader):
                # loader 里面的数据已经进过StandardScale，将全部数据转换为正态分布了。
                batch_x = batch_x.float().to(self.args.device)
                batch_x_stamp = batch_x_stamp.to(self.args.device)
                batch_y = batch_y.to(self.args.device)
                batch_y_stamp = batch_y_stamp.to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                output= self.model(batch_x,batch_x_stamp,dec_inp,batch_y_stamp)

                output = output[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                # 'Tensor' object has no attribute 'append'

                output = output.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred.append(output)
                true.append(batch_y)

            pred = np.array(pred)
            true = np.array(true)

            pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
            true = true.reshape(-1, true.shape[-2], true.shape[-1])

            MSE, MAE, MAPE, MSPE,RMSE = metric(pred, true)
            print('MSE:{}, MAE:{}'.format(MSE, MAE))
            macs, params = profile(self.model, inputs=inputs)
            print(f"MACs: {macs}, Params: {params}")
            if args.save_results is True:
                save_results(args, MSE, MAE, macs, params, 'results.txt')

    def train(self):
        # 1.加载数据集，打印数据集长度，保存训练开始时间

        train_start_time = time.time()
        train_data,train_loader = self._get_data(self.args,'train',shuffle=True)
        val_data,val_loader = self._get_data(self.args,'val',shuffle=False)
        test_data,test_loader = self._get_data(self.args,'test',shuffle=False)
        print('train:',len(train_data),'val:',len(val_data),'test:',len(test_data))
        total_steps = len(train_data)// self.args.batch_size

        x = torch.randn(16,96,7).to(self.device)
        y = torch.randn(16, 144,7).to(self.device)
        x_stamp = torch.randn(16,96,4).to(self.device)
        y_stamp = torch.randn(16, 144,4).to(self.device)

        # 2.设置优化器，损失函数，回调函数（早停，调整学习率，保存检查点）init已经实现过了
        checkpoint_path = self.args.checkpoint_path
        early_stopping = EarlyStopping(args)
        # 3.设置循环的epoch，初始化每一个epoch的loss，设置epoch开始时间
        for epoch in range(args.epoch):
            train_loss = []
            self.model.train()
            epoch_start_time = time.time()
            # 4.取出batch循环，step = len(data)/batch_size,数据放到GPU上，清零每个batch的梯度
            # enumerate:枚举，可以返回每个批次的索引i
            # batch_x,batch_x_stamp,batch_y,batch_y_stamp要加括号
            with tqdm(total=total_steps, desc=f'Epoch {epoch + 1}/{args.epoch}', unit='step',dynamic_ncols=True) as pbar:
                for i,(batch_x,batch_x_stamp,batch_y,batch_y_stamp) in  enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_stamp = batch_x_stamp.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_y_stamp = batch_y_stamp.float().to(self.device)

                    # 注意不能把batch_y直接丢进decoder,要把预测的部分赋0
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    self.optimizer.zero_grad()
                    output= self.model(batch_x,batch_x_stamp,dec_inp,batch_y_stamp)
                    # print(periods)
                    output = output[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.criterion(output,batch_y)
                    pbar.update(1)
                    train_loss.append(loss.item())
                    # loss.backward() 用于计算梯度，而 model_optim.step() 则根据这些梯度更新模型参数
                    loss.backward()
                    self.optimizer.step()
            vali_loss = self.vali(val_loader)
            # 这里的测试集相当于验证集了
            test_loss = self.vali(test_loader)
            print("Epoch: {} cost time: {:.4f} \n "
                  "train_loss: {:.4f} vali_loss: {:.4f} test_loss: {:.4f}".format(epoch+1, time.time() - epoch_start_time, np.average(train_loss), vali_loss, test_loss))
            early_stopping(vali_loss, model=self.model, path=checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(args, epoch, optimizer=self.optimizer)
            best_model_path = os.path.join(checkpoint_path, 'ckpt_best.pth')
    def vali(self, vali_loader):
        # 设置模型为评估模式
        # 禁用 Dropout：在评估模式下，dropout 层不再随机丢弃神经元，从而确保每个神经元在推理时都参与计算。
        # Batch Normalization：批归一化层在评估模式下使用训练期间计算的全局均值和方差，而不是使用当前批次的均值和方差。
        # 在评估模式下，模型的输出是基于固定的参数和输入的，避免了训练过程中引入的随机性（如 dropout 和 batch normalization 中的动态行为）。这使得每次评估的结果可重复，便于比较不同模型或超参数设置的效果。
        self.model.eval()
        val_loss = []
        # 不更计算梯度
        with torch.no_grad():
            for i,(batch_x,batch_x_stamp,batch_y,batch_y_stamp) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                output= self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp)
                output= output[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                # 与训练不同，求完loss之后不需要更新梯度
                loss = self.criterion(output,batch_y)
                val_loss.append(loss.item())
        #重新设置模型为训练模式
        self.model.train()
        return np.average(val_loss)

# 使用参数
if __name__ == "__main__":
    args = get_args()
    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
    exp = Long_term_Forcast(args)
    print('>>>>>>>model parameter>>>>>>>>>>>>>>>>>>>>>>>>>>')
    params = [(key, getattr(args, key)) for key in dir(args)
              if not key.startswith('__') and not callable(getattr(args, key))]

    # 使用 tabulate 打印单列表格
    print(tabulate(params, headers=['Parameter', 'Value'], colalign=("left", "center"), tablefmt='fancy_grid'))
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    time_train_start = time.time()
    exp.train()
    time_train_end = time.time()
    print('>>>>total training time:', time_train_end - time_train_start)
    print('>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.test(args)




