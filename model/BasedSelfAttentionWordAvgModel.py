import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging as logger


class BasedSelfAttentionWordAvgModel(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, padding_idx, dropout, best_model_path, initrange=0.1):
        super(BasedSelfAttentionWordAvgModel, self).__init__()
        self.best_model_path = best_model_path
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_size, output_size)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.init_weight(initrange)

    def forward(self, text):
        embed = self.dropout(self.embed(text))

        embed = embed.permute(1, 0, 2)

        q = self.query(embed)

        k = self.key(embed)

        v = self.value(embed)

        atte = self.attention(q, k, v)

        h_self = torch.sum(atte, dim=1).squeeze()

        return self.linear(h_self)

    def attention(self, q, k, v):
        d_k = k.size(-1)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        score_softmax = F.softmax(score, dim=-1)

        atte = torch.matmul(score_softmax, v)
        return atte

    def init_weight(self, initrange):
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def save_model(self):
        """
        保存最优模型
        :return:
        """
        torch.save(self.state_dict(), self.best_model_path)

    def init_pre_train(self, pretrain_vector, pad_idx, unk_idx):
        self.embed.weight.data.copy_(pretrain_vector)
        self.embed.weight.data[pad_idx] = torch.zeros(self.embed_size)
        self.embed.weight.data[unk_idx] = torch.zeros(self.embed_size)

