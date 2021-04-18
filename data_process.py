import torch
import torchtext


class DataProcessor(object):
    """
    数据处理
    """

    def __init__(self, root, train_path, validation_path, test_path, format="csv"):
        self.TEXT = torchtext.data.Field(lower=True)
        self.LABEL = torchtext.data.LabelField(dtype=torch.float)
        self.root = root
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.format = format

        self.train_data, self.val_data, self.test_data \
            = torchtext.data.TabularDataset.splits(path=self.root,
                                                   train=self.train_path,
                                                   validation=self.validation_path,
                                                   test=self.test_path,
                                                   format=self.format,
                                                   fields=[('Text', self.TEXT), ('Label', self.LABEL)])

    def build_vocab(self, vocab_size, vector):
        """
        构建词表
        :param vocab_size:词表大小
        :param vector: 词向量
        :return:TEXT.vocab, LABEL.vocab
        """
        self.TEXT.build_vocab(self.train_data, max_size=vocab_size, vectors=vector, unk_init=torch.Tensor.normal_)
        self.LABEL.build_vocab(self.train_data)
        return self.TEXT.vocab, self.LABEL.vocab

    def get_len(self):
        """
        :return: 一个词表大小
        """
        return len(self.TEXT.vocab.stoi)

    def get_vocab_stoi(self):
        """
        :return: 词典stoi的映射，类型是一个dict
        """
        return self.TEXT.vocab.stoi

    def get_vocab_itos(self):
        """
        :return: 词典的itos, 类型是一个数组
        """
        return self.TEXT.vocab.itos

    def get_word_index(self, word):
        """
        给定一个单词，返回单词在词典中的下标
        :param word: 单词
        :return: 单词在词典中的下标
        """
        return self.TEXT.vocab.stoi[word]

    def give_index_to_word(self, index):
        """
        给定一个下标，返回此下标对应的单词
        :param index: 下标
        :return:此下标对应的单词
        """
        return self.TEXT.vocab.itos[index]

    def get_unk_idx(self):
        """

        :return: "<unk>" 再词典中的下标
        """
        return self.TEXT.vocab.stoi["<unk>"]

    def get_pad_idx(self):
        """

        :return: "<pad>" 再词典中的下标
        """
        return self.TEXT.vocab.stoi["<pad>"]

    def get_pretrain_vector(self):
        """

        :return: 构建词典是使用的预训练词向量
        """
        return self.TEXT.vocab.vectors

    def get_data(self, batch_size, sort_key, device, shuffle=True):
        """
        :param batch_size:batch_size
        :param shuffle: shuffle, 默认为True
        :param sort_key:
        :param device:
        :return: 返回train_iter, val_iter, test_iter
        """
        train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
            (self.train_data,
             self.val_data,
             self.test_data),
            batch_size=batch_size,
            shuffle=shuffle,
            sort_key=sort_key,
            device=device)
        return train_iter, val_iter, test_iter
