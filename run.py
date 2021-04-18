from data_process import DataProcessor
from model.config import TrainConfig
from data_config import DataConfig
from model.WordAvgModel import WordAvgModel
from model.BasedAttentionWordAvgModel import BasedAttentionWordAvgModel
from model.BasedSelfAttentionWordAvgModel import BasedSelfAttentionWordAvgModel
import torch
import torch.nn as nn
from train import ModelTrainer
from excutor import ModelExecutor
import logging as logger
from utils.common_util import (
    is_use_cuda,
    init_seed,
    get_device,
    poly_lr_scheduler,
    record_train_model
)
from typing import Tuple
from utils.log_util import Logger
import time
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from pretrain.train import BertTrainer
import pretrain.dataset


init_seed(seed=2021)

data_config = DataConfig(format="tsv")

data_processor = DataProcessor(root=data_config.root,
                               train_path=data_config.train_path,
                               validation_path=data_config.validation_path,
                               test_path=data_config.test_path,
                               format=data_config.format)

model_config = TrainConfig()
# 可使用model_config.reset_config 自定义模型参数或者直接修改config文件
model_config.reset_config(my_cuda_is_enable=False, model_name='bert-classification')
if model_config.model_name is None:
    raise ValueError("Invalid model-name value: {}".format(model_config.model_name))

data_processor.build_vocab(vocab_size=model_config.vocab_size, vector="glove.6B.100d")

train_iter, val_iter, test_iter = data_processor.get_data(batch_size=model_config.batch_size,
                                                          shuffle=True,
                                                          sort_key=lambda x: len(x.Text),
                                                          device=get_device())

PAD_IDX = data_processor.get_pad_idx()
UNK_IDX = data_processor.get_unk_idx()
pretrain_vector = data_processor.get_pretrain_vector()
vocab_size = data_processor.get_len()
if model_config.model_name == 'word-avg':
    wordavg_model = WordAvgModel(vocab_size=vocab_size,
                                 embed_size=model_config.embed_size,
                                 output_size=1,
                                 padding_idx=PAD_IDX,
                                 dropout=model_config.dropout,
                                 best_model_path="./pth/best-avg-model.pth").to(get_device())

    wordavg_model.init_pre_train(pretrain_vector=pretrain_vector, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    avg_optim = torch.optim.Adam(wordavg_model.parameters(), lr=model_config.lr)
    avg_loss_fn = nn.BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=avg_optim, lr_lambda=poly_lr_scheduler)
    trainer = ModelTrainer(model=wordavg_model, loss_fn=avg_loss_fn, optimizers=(avg_optim, None))

    logger.info("Word Averaging Model")
    executor = ModelExecutor(model=wordavg_model,
                             trainer=trainer,
                             train_data=train_iter,
                             evaluate_data=val_iter)
    best_acc = executor.execute(model_config.epoch_size)
    logger.info(f'当前模型最优情况下的验证准确率： {best_acc * 100:.2f}%')

elif model_config.model_name == 'attention-word-avg':
    atte_wordavg_model = BasedAttentionWordAvgModel(vocab_size=vocab_size,
                                                    embed_size=model_config.embed_size,
                                                    output_size=1,
                                                    padding_idx=PAD_IDX,
                                                    dropout=model_config.dropout,
                                                    best_model_path="./pth/best-atte-avg-model.pth").to(get_device())

    atte_wordavg_model.init_pre_train(pretrain_vector=pretrain_vector, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    atte_avg_optim = torch.optim.Adam(atte_wordavg_model.parameters(), lr=model_config.lr)
    atte_avg_loss_fn = nn.BCEWithLogitsLoss()

    record_train_model(model_name="WordAvgModel", config=model_config, train_result=best_acc)
    # atte_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=avg_optim, lr_lambda=poly_lr_scheduler)
    atte_trainer = ModelTrainer(model=atte_wordavg_model,
                                optimizers=(atte_avg_optim, None),
                                loss_fn=atte_avg_loss_fn)

    logger.info("Based Attention Word Averaging Model")
    executor = ModelExecutor(model=atte_wordavg_model,
                             trainer=atte_trainer,
                             train_data=train_iter,
                             evaluate_data=val_iter)

    best_model_acc = executor.execute(model_config.epoch_size)
    logger.info(f'当前模型最优情况下的验证准确率：{best_model_acc * 100:.2f}%')
    record_train_model(model_name="BasedAttentionWordAvgModel", config=model_config, train_result=best_model_acc)

elif model_config.model_name == 'self-attention-word-avg':
    '''
        基于自注意力机制的WAVG 模型
    '''
    selfatte_wordavg_model = BasedSelfAttentionWordAvgModel(vocab_size=vocab_size,
                                                            embed_size=model_config.embed_size,
                                                            output_size=1,
                                                            padding_idx=PAD_IDX,
                                                            dropout=model_config.dropout,
                                                            best_model_path="./pth/best-selfatte-model.pth").to(get_device())

    selfatte_wordavg_model.init_pre_train(pretrain_vector=pretrain_vector, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    avg_optim = torch.optim.Adam(selfatte_wordavg_model.parameters(), lr=model_config.lr)
    avg_loss_fn = nn.BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=avg_optim, lr_lambda=poly_lr_scheduler)
    trainer = ModelTrainer(model=selfatte_wordavg_model, loss_fn=avg_loss_fn, optimizers=(avg_optim, None))

    logger.info("Based Self-Attention WordAvg Model")
    executor = ModelExecutor(model=selfatte_wordavg_model,
                             trainer=trainer,
                             train_data=train_iter,
                             evaluate_data=val_iter)

    selfatte_best_acc = executor.execute(model_config.epoch_size)
    logger.info(f'当前模型最优情况下的验证准确率： {selfatte_best_acc * 100:.2f}%')
    record_train_model(model_name="BasedSelfAttentionWordAvgModel", config=model_config, train_result=selfatte_best_acc)

elif model_config.model_name == 'bert-classification':
    '''
    Based BERT
    '''
    bert_config = BertConfig.from_pretrained(model_config.bert_path)
    bert_model = BertForSequenceClassification.from_pretrained(model_config.bert_path, config=bert_config)
    bert_tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)
    train_dataset = pretrain.dataset.SentiDataSet(root=data_config.root,
                                                  path=data_config.train_path,
                                                  tokenizer=bert_tokenizer,
                                                  logger=logger,
                                                  label_vocab_path=data_config.label_vocab_path)

    valid_dataset = pretrain.dataset.SentiDataSet(root=data_config.root,
                                                  path=data_config.validation_path,
                                                  tokenizer=bert_tokenizer,
                                                  logger=logger,
                                                  label_vocab_path=data_config.label_vocab_path)

    bert_trainer = BertTrainer(args=model_config,
                               model=bert_model,
                               tokenizer=bert_tokenizer,
                               train_dataset=train_dataset,
                               val_dataset=valid_dataset)

    bert_executor = ModelExecutor(args=model_config, model=bert_model, trainer=bert_trainer)
    bert_executor.execute(model_config.epoch_size)

