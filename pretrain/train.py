from utils.common_util import (
    get_device,
    calculate_acc
)
from pretrain.dataset import PadBatchCollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pretrain.optim import BertAdam, NoamOpt
import logging as logger
from torch.utils.tensorboard import SummaryWriter
import os


class BertTrainer:
    def __init__(self, args, model, tokenizer, train_dataset, val_dataset, device=get_device()):

        self.config = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.model = model.to(self.device)
        base_optimizer = BertAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optim = NoamOpt(self.config.embed_size, 0.1, self.config.lr_warmup, base_optimizer)
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           collate_fn=PadBatchCollateFn(self.tokenizer.pad_token_id),
                                           shuffle=True,
                                           pin_memory=True)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=PadBatchCollateFn(self.tokenizer.pad_token_id),
                                         shuffle=True,
                                         pin_memory=True)

        self.train_writer = SummaryWriter(os.path.join(args.log_dir, 'train_cls'))
        self.valid_writer = SummaryWriter(os.path.join(args.log_dir, 'valid_cls'))

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _train(self, epoch):
        self.model.train()
        loss, acc, step_count = 0., 0., 0
        total = len(self.train_dataloader)
        for i, batch_data in enumerate(self.train_dataloader):
            text, label = batch_data['sent_token'].to(self.device), batch_data['label'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            bert_output = self.model(text, attention_mask=attention_mask, return_dict=True)
            batch_loss = self.criterion(bert_output.logits, label)

            batch_acc = (torch.argmax(bert_output.logits, dim=1) == label).float().mean()

            '''
                trick: 当数据维度过大时，我们可以再不改变模型参数的条件下， 将一个batch 拆成若干步骤，一次进行反向传播，使得的计算时所需内存大幅下降
                    以时间换空间的思路
            '''
            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()
            loss += batch_loss.item()
            acc += batch_acc.item()
            step_count += 1
            curr_step = self.optim.curr_step()

            lr = self.optim.param_groups[0]['lr']

            if (i + 1) % self.config.batch_split == 0:
                self.optim.step()
                self.optim.zero_grad()

                loss /= step_count
                acc /= step_count
                self.train_writer.add_scalar('ind/loss', loss, curr_step)
                self.train_writer.add_scalar('ind/acc', acc, curr_step)
                self.train_writer.add_scalar('ind/lr', lr, curr_step)
                logger.info(f'\tEpoch: {epoch} | step: {curr_step}')
                logger.info(f'\tTrain Loss: {loss:.3f} | Train Acc: {acc * 100:.2f}%')
                loss, acc, step_count = 0, 0, 0
                if curr_step % self.config.eval_steps == 0:
                    self._eval_train(curr_step)

    def _eval_train(self, curr_step):
        self.model.eval()
        with torch.no_grad():
            all_logits = []
            all_label = []
            for i, batch_data in enumerate(self.val_dataloader):
                text, label = batch_data['sent_token'].to(self.device), batch_data['label'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                val_output = self.model(text, attention_mask=attention_mask, return_dict=True)
                all_label.append(label)
                all_logits.append(val_output.logits)

            all_label = torch.cat(all_label, dim=0)
            all_logits = torch.cat(all_logits, dim=0)

            val_loss = self.criterion(all_logits, all_label).float()
            val_acc = (torch.argmax(all_logits, dim=1) == all_label).float().mean()
            self.valid_writer.add_scalar('ind/loss', val_loss, curr_step)
            self.valid_writer.add_scalar('ind/acc', val_acc, curr_step)
            logger.info(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')
        self.model.train()

    def train(self, start_epoch, epoch_size, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epoch_size):
            logger.info('Training on epoch'.format(epoch))
            self._train(epoch)
            for func in after_epoch_funcs:
                func(epoch, self.device)