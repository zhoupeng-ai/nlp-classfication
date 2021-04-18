import logging as logger
from utils.common_util import EarlyStopping
import torch
import os
from pretrain.train import BertTrainer



class ModelExecutor(object):
    def __init__(self, args, model, trainer, train_data=None, evaluate_data=None):
        """
        模型的执行器
        :param model: 模型
        :param trainer: 训练器
        :param train_data: 训练数据
        :param evaluator: 验证器
        :param evaluate_data: 验证数据
        """
        self.config = args
        self.model = model
        self.trainer = trainer
        self.train_data = train_data
        self.evaluate_data = evaluate_data
        if isinstance(trainer, BertTrainer):
            self.is_bert = True

    def execute(self, epoch_size):
        """
        模型开始执行后，我们会在每次训练时都在验证集上进行验证，评估准确率，并保存下最好的模型
        :param epoch_size: 迭代次数
        :return:
        """
        best_model_acc = 0.
        logger.info(f'{"=" * 20}开始训练 {"=" * 20}')
        if self.is_bert:
            start_epoch = 0
            self.trainer.train(start_epoch, epoch_size, after_epoch_funcs=[self.save_func])
            return

        # 准确率连降10次则停止训练
        early_stop = EarlyStopping(patience=10)
        for epoch in range(epoch_size):
            logger.info(f'epoch: {epoch}, --start')
            train_acc, train_loss = self.trainer.train(self.train_data)
            val_acc, val_loss = self.trainer.evaluate(self.evaluate_data)
            early_stop.early_stopping_acc(val_acc)

            if early_stop.early_stop:
                logger.info("Early stopping")
                break

            logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            logger.info(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

            if val_acc > best_model_acc:
                best_model_acc = val_acc
                self.model.save_model()
                logger.info("Save the best model!")
            logger.info(f'epoch: {epoch}, --end')
        logger.info(f'{"=" * 20}训练结束 {"=" * 20}')

        return best_model_acc

    def get_ckpt_filename(self, name, epoch):
        return '{}-{}.pth'.format(name, epoch)

    def save_func(self, epoch, device):
        filename = self.get_ckpt_filename('bert_model', epoch)
        torch.save(self.trainer.state_dict(), os.path.join(self.config.model_root_path, filename))


