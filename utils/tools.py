import math
import os
from datetime import datetime
import torch


class EarlyStopping(object):
    def __init__(self,args,delta = 0.001):
        self.best_score = None
        self.args = args
        self.patience = args.patience
        self.early_stop = False
        self.counter = 0
        self.delta = delta
        self.verbose = args.verbose #有的时候不需要显示调试信息
    def __call__(self,val_loss,model,path):
        # 在一些代码库和研究工作中，负损失的最大化是常见做法。
        # 这可能源于很多研究人员使用了类似的基准代码，所以习惯了用负数的方式
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score > self.best_score+self.delta: #浮点数比绝对大小不合适
            self.best_score = score
            self.save_checkpoint(model,path)
            self.counter = 0 #counter归零
        else:
            self.counter += 1

            if self.verbose:
                print('Early stopping counter: {} \n val_loss increase: {:.4f} --> {:.4f}'.format(self.counter, -self.best_score, val_loss))
            if self.counter >= self.patience:
                self.early_stop = True
    def save_checkpoint(self, model,path):
        # 添加地址最好使用os.path.join
        filepath = os.path.join(path,self.args.model+'_'+ self.args.data+'_' + str(self.args.seq_len)+'_'+str(self.args.pred_len)+'_'+'ckpt_best.pth')
        torch.save(model.state_dict(),filepath)
        if self.verbose:
            print('Saving checkpoint to',filepath)

def adjust_learning_rate(args,epoch,optimizer):
    if args.adjust_lr_type == 'exponential_decay':
        epoch_lr = {epoch:args.lr * (0.5**epoch) }
    elif args.adjust_lr_type == 'cosine_decay':
        epoch_lr = {epoch:args.lr / 2 * (math.cos(epoch/args.epoch * math.pi)+1)}
    elif args.adjust_lr_type == 'Sparse':
        epoch_lr = {epoch:args.lr if epoch < 3 else args.lr *  (0.8 ** ((epoch - 3) // 1))}
    elif args.adjust_lr_type == 'type3':
        epoch_lr = {epoch: args.lr * (0.95 ** ((epoch - 1) // 1))}
    if epoch in epoch_lr.keys():
        lr = epoch_lr[epoch]
        # betas:Adam 优化器使用一阶和二阶动量估计来调整学习率。betas 是一个元组 (beta1, beta2)，分别表示一阶和二阶动量的衰减率。
        #eps (epsilon):用于数值稳定性的小常数，防止除零错误。
        #amsgrad:AMSGrad 是一种 Adam 优化器的变体，通过保持历史梯度的最大值来防止梯度的爆炸问题，从而提高收敛的稳定性。
        #maximize:布尔值，指定优化器是否要最大化目标函数，而不是最小化。
        # foreach:开启 foreach 后，优化器会使用批量操作进行参数更新，这通常可以提高大型模型或多 GPU 训练的性能
        # capturable:如果设为 True，优化器可以与 CUDA 图配合使用，以便更高效的 GPU 训练。
        # differentiable:如果设为 True，则优化器的更新步骤允许通过自动微分计算梯度，这在一些高级算法（如元学习）中很有用
        # fused:融合实现（Fused Implementation）使用特定的 CUDA 核函数来加速优化器更新步骤，尤其适合在 NVIDIA GPU 上训练大模型。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Adjusted learning rate:', lr)


import os
from datetime import datetime


def save_results(args, mse, mae, macs, params, file_name):
    # 准备数据，包含你想保存的 `args` 参数
    result = {
        'Model': args.model,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'MSE': mse,
        'MAE': mae,
        'lr': args.lr
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 定义表头和格式化的行输出
    header = "{:<25} {:<15} {:<15} {:<15} {:<15}  {:<15} {:<15} \n".format(
        'Time', 'Model', 'Data', 'seq_len', 'pred_len', 'MSE', 'MAE'
    )

    row = "{:<25} {:<15} {:<15} {:<15} {:<15} {:<15,.4f} {:<15,.4f}  \n".format(
        timestamp, args.model, args.Data, args.seq_len, args.pred_len, mse, mae
    )

    custom_message = f"实验简介:\n"

    # 如果文件存在，先读取原始内容
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            original_content = f.read()  # 读取原始内容
    else:
        original_content = ""  # 文件不存在时，原始内容为空

    # 生成新的内容，将其拼接在原始内容之前
    new_content = custom_message + header + row + "\n" + original_content

    # 写入文件，覆盖原内容
    with open(file_name, 'w') as f:
        f.write(new_content)





