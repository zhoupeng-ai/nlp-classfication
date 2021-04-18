import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common_util import (
    get_device
)


class BasedAttentionWordAvgModel(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, padding_idx, dropout,  best_model_path, initrange=0.1):
        super(BasedAttentionWordAvgModel, self).__init__()
        self.best_model_path = best_model_path
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.u = nn.Parameter(torch.rand(embed_size))
        self.linear = nn.Linear(embed_size, output_size)
        self.init_weight(initrange)
        self.embed_size = embed_size

    def forward(self, text):
        # text.shape: [seq_len, batch_size]
        # mask.shape: [seq_len, batch_size]
        # embed.shape: [seq_len, batch_size, embed_size]
        embed = self.dropout(self.embed(text))
        # 计算余弦距离
        # x1: u.shape : [embed_size] --- > [1, embed_size]
        # x2: embed.shape:[seq_len, batch_size, embed_size] -- > [seq_len*batch_size, embed_size]
        # cos_sim.shape: [seq_len*batch_size, embed_size]
        # alpha_t.shape: [seq_len, batch_size]
        alpha_t = torch.exp(F.cosine_similarity(self.u.view(1, -1).to(get_device()), embed.view(-1, embed.shape[2]), dim=1)).view(embed.shape[0], embed.shape[1])
        # h_t = alpha_t * embed
        h_t = alpha_t.unsqueeze(2).repeat(1, 1, embed.shape[2]) * embed
        pooled = self.dropout(h_t)

        mask = self.mask(text).unsqueeze(2)
        mask_embed = pooled * mask
        masked = mask_embed.sum(0)

        return self.linear(masked)

    def init_weight(self, initrange):
        """
        初始化权重
        :param initrange:
        :return:
        """
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

    def mask(self, text):
        return (text != 0).float().to(get_device())

