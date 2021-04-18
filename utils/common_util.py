import logging as logger
import numpy as np


def is_use_cuda():
    """
        判断GPU是否可用
    """
    import torch

    return torch.cuda.is_available()


def get_device():
    """
        获取GPU或者CPU
    """
    return "cuda" if is_use_cuda() else "cpu"


def init_seed(seed):
    """
       设定随机种子，保证模型课复现
    """
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if is_use_cuda():
        torch.cuda.manual_seed(seed)


def calculate_acc(pred_y, real_y):
    import torch

    round_y = torch.round(torch.sigmoid(pred_y))

    correct = (round_y == real_y).float()

    acc = correct.sum() / len(correct)

    return acc


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch / num_epochs) ** power


def record_train_model(model_name, config, train_result):
    import time
    import json
    import inspect
    curr_day = time.strftime("%Y-%m-%d", time.localtime())
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    params = inspect.signature(config.__init__).parameters
    model_process_params = {}
    model_param = {}
    train_params = {}
    train_result = train_result if train_result else 0.
    for param in params:
        param = str(params[param]).split("=")
        train_params[param[0]] = param[1]

    model_param[model_name if model_name else "DEFAULT"] = {"train-best-acc": f'{train_result * 100:.4f}%',
                                                            "train-params": train_params}

    model_process_params[curr_day] = {"train-time": curr_time,
                                      "train-model": model_param
                                      }

    model_process_params_json = json.dumps(model_process_params)

    with open('./log/model_train_result.json', 'a') as file:
        file.write(model_process_params_json)
        file.write("\n")


def load_vocab(vocab_file):
    with open(vocab_file) as f:
        res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))  # list, token2index, index2token


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def early_stopping_loss(self, val_loss):
        score = -val_loss
        if self.best_loss is None:
            self.best_loss = score
        elif score < self.best_loss:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.counter = 0
        return self.early_stop

    def early_stopping_acc(self, val_acc):
        score = val_acc
        if self.best_acc is None:
            self.best_acc = score
        elif score < self.best_acc:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = score
            self.counter = 0
        return self.early_stop
