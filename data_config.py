class DataConfig(object):
    def __init__(self,
                 root="./data",
                 train_path="senti.train.tsv",
                 validation_path="senti.dev.tsv",
                 test_path="senti.test.tsv",
                 label_vocab_path="label_vocab.txt",
                 format="csv"):
        """

        :param root: 数据根路径
        :param train_path: 训练数据文件
        :param validation_path: 验证数据文件
        :param test_path: 测试数据文件
        :param format: 数据文件文件格式, 缺省默认csv
        """
        self.root = root
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.format = format
        self.label_vocab_path = label_vocab_path

