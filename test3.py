class Long_term_Forcast():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = self._build_model().to(args.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def find_learning_rate(self, train_loader, start_lr=1e-8, end_lr=10, num_iter=100):
        """
        执行学习率范围测试，观察损失如何随学习率变化。
        Args:
            train_loader: 训练数据加载器
            start_lr: 初始学习率
            end_lr: 最大学习率
            num_iter: 测试的迭代次数
        Returns:
            一个字典，包含学习率和损失列表。
        """
        # 记录学习率和损失值
        lr_log = []
        loss_log = []

        # 设置学习率按指数递增
        lr_lambda = lambda x: start_lr * (end_lr / start_lr) ** (x / num_iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        self.model.train()
        iterator = iter(train_loader)

        for i in range(num_iter):
            try:
                batch_x, batch_x_stamp, batch_y, batch_y_stamp = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch_x, batch_x_stamp, batch_y, batch_y_stamp = next(iterator)

            batch_x = batch_x.float().to(self.device)
            batch_x_stamp = batch_x_stamp.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_y_stamp = batch_y_stamp.float().to(self.device)

            # 创建 Decoder 输入
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # 计算输出与损失
            self.optimizer.zero_grad()
            output = self.model(batch_x, batch_x_stamp, dec_inp, batch_y_stamp)
            output = output[:, -self.args.pred_len:, :]
            loss = self.criterion(output, batch_y[:, -self.args.pred_len:, :])
            loss.backward()
            self.optimizer.step()

            # 更新学习率
            current_lr = lr_lambda(i)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # 记录损失和学习率
            lr_log.append(current_lr)
            loss_log.append(loss.item())

        return {'lr': lr_log, 'loss': loss_log}

# 在训练开始前调用学习率范围测试
exp = Long_term_Forcast(args)
train_data, train_loader = exp._get_data(args, 'train', shuffle=True)
result = exp.find_learning_rate(train_loader)

# 绘制损失 vs. 学习率曲线
plt.plot(result['lr'], result['loss'])
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.show()
