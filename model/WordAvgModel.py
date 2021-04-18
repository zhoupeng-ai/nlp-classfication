import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAvgModel(nn.Module):
    """
    基于Word Average 的模型， 基本思想是利用词平均来输出句子向量
    """
    def __init__(self, vocab_size, embed_size, output_size, padding_idx, dropout, initrange=0.1, best_model_path=None):
        """

        :param vocab_size:词表大小
        :param embed_size: 词嵌入矩阵维度
        :param output_size: 输出矩阵维度
        :param padding_idx: pad index
        :param dropout: dropout
        :param initrange: 初始化参数范围(-initrange, initrange)
        """
        super(WordAvgModel, self).__init__()
        self.embed_size = embed_size
        self.best_model_path = best_model_path
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx= padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_size, output_size)
        self.init_weight(initrange)

    def forward(self, text):
        """

        :param text: 供模型学习训练的文本， shape[seq_len, batch_size]
        :return:
        """
        # text.shape [seq_len, batch_size]
        # embed.shape : [seq_len, batch_size, embed_size]
        embed = self.dropout(self.embed(text))
        # [batch_size, seq_len, embed_size]
        embed = embed.permute(1, 0, 2)
        pooled = F.avg_pool2d(embed, kernel_size=(embed.shape[1], 1), stride=1)
        # pooled.shape: [batch_size, 1, embed_size]
        pooled = pooled.squeeze(1)
        # output.shape: [batch_size, 1, output_size]
        output = self.linear(pooled)
        return output

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

