from utils.common_util import (
    get_device,
    calculate_acc
)
import torch
from typing import Tuple


class ModelTrainer(object):
    """
    模型训练器
    """
    def __init__(self, model, loss_fn, optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
        """
        模型的构造方法
        :param model: 模型
        :param optimizers: 优化器
        :param loss_fn: 损失函数
        """
        self.model = model.to(get_device())
        self.optimizers = optimizers
        self.loss_fn = loss_fn.to(get_device())

    def train(self, train_data):
        """
        模型训练函数
        :param train_data: 训练数据
        :return:
        """
        epoch_acc = 0.
        epoch_loss = 0.
        total_len = 0
        self.model.train()
        for i, batch in enumerate(train_data):
            self.optimizers[0].zero_grad()
            pred_y = self.model(batch.Text).squeeze(1)
            train_acc = calculate_acc(pred_y=pred_y, real_y=batch.Label)
            train_loss = self.loss_fn(pred_y, batch.Label)
            train_loss.backward()

            self.optimizers[0].step()
            epoch_acc += train_acc * len(batch.Label)
            epoch_loss += train_loss * len(batch.Label)
            total_len += len(batch.Label)

        return epoch_acc / total_len, epoch_loss / total_len

    def evaluate(self, validation_data, is_mask=False):
        """
        模型训练函数
        :param validation_data: 训练数据
        :param is_mask: 是否需要进行遮蔽训练
        :return:
        """
        epoch_acc = 0.
        epoch_loss = 0.
        total_len = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(validation_data):
                pred_y = self.model(batch.Text).squeeze(1)
                val_acc = calculate_acc(pred_y=pred_y, real_y=batch.Label)
                val_loss = self.loss_fn(pred_y, batch.Label)
                epoch_acc += val_acc * len(batch.Label)
                epoch_loss += val_loss * len(batch.Label)
                total_len += len(batch.Label)
        self.model.train()
        return epoch_acc / total_len, epoch_loss / total_len


