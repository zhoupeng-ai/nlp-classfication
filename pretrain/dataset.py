import torch
import torch.utils.data as tud
import os
from utils.common_util import load_vocab


class SentiDataSet(tud.Dataset):
    def __init__(self, root, path, tokenizer, label_vocab_path, logger, max_seq_length=2048):
        super(SentiDataSet, self).__init__()
        self.logger = logger
        self.path = os.path.join(root, path)
        self.label_vocab_path = os.path.join(root, label_vocab_path)
        self.tokenizer = tokenizer
        _, self.label_vocab, _ = load_vocab(self.label_vocab_path)
        self.max_seq_length = max_seq_length
        self.data = SentiDataSet.make_data(self)

    @staticmethod
    def make_data(self):
        dataset = []
        self.logger.info("Loading SentiDataSet")
        self.logger.info(f"Reading data from {self.path}")
        with open(self.path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if len(line.strip()) != 0]
            label_text = [line.split("\t") for line in lines]
            for text, label in label_text:
                '''
                bertTokenizer(sentences)
                bert 的分词器， 基于WordPiece 分词， 分词之后的结构为
                    输出结构为    字典 {
                                        input_ids:'[]' 表示文本
                                        token_type_ids:'', 表示文本中的句子
                                        attention_mask:''  表示
                                    }
                '''
                sent_token = self.tokenizer(text[:self.max_seq_length])
                dataset.append([int(self.label_vocab[label]),
                                sent_token["input_ids"],
                                sent_token["attention_mask"]])
        self.logger.info(f"{len(dataset)} data from {self.path} loaded")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, sent_token, attention_mask = self.data[idx]
        return {"label": label, "sent_token": sent_token, "attention_mask": attention_mask}


class PadBatchCollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        res = dict()
        res['label'] = torch.tensor([i['label'] for i in batch]).long()
        max_len = max([len(i['sent_token']) for i in batch])
        res['sent_token'] = torch.tensor([i['sent_token'] +
                                         [self.pad_idx] * (max_len - len(i['sent_token']))
                                         for i in batch]).long()

        res['attention_mask'] = torch.tensor([i['attention_mask'] +
                                              [self.pad_idx] * (max_len - len(i['attention_mask']))
                                              for i in batch]).long()
        return res
