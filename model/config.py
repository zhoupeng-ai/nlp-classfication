class TrainConfig:
    def __init__(self,
                 vocab_size=20000,
                 embed_size=100,
                 batch_size=128,
                 epoch_size=100,
                 lr=1e-2,
                 dropout=0.5,
                 bert_path="bert-base-uncased",
                 model_root_path="./pth",
                 log_dir="./log",
                 eval_steps=2,
                 lr_warmup=100,
                 batch_split=3,
                 weight_decay=0.01,
                 model_name='word-avg'):
        """
        模型参数配置
        :param vocab_size:
        :param embed_size:
        :param batch_size:
        :param epoch_size:
        :param lr:
        :param dropout:
        """
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.lr = lr
        self.dropout = dropout
        self.bert_path = bert_path
        self.model_root_path = model_root_path
        self.eval_steps = eval_steps
        self.lr_warmup = lr_warmup
        self.batch_split = batch_split
        self.weight_decay = weight_decay
        self.log_dir = log_dir
        self.model_name = model_name

    def reset_config(self,
                     my_cuda_is_enable=False,
                     vocab_size=20000,
                     embed_size=100,
                     batch_size=32,
                     epoch_size=20,
                     lr=1e-2,
                     dropout=0.5,
                     eval_steps=2,
                     lr_warmup=100,
                     batch_split=3,
                     weight_decay=0.01,
                     model_name='word-avg'):
        """
            本地无法支持复杂模型，降低参数量
        :param my_cuda_is_enable: default False
        :param vocab_size:20000
        :param embed_size:100
        :param batch_size:32
        :param epoch_size:20
        :param lr:1e-2
        :param dropout:0.5
        :param eval_steps:2,
        :param lr_warmup:200,
        :param batch_split:3
        :param weight_decay:0.01
        :param model_name: 默认word-avg, \
                    可选值【word-avg: 使用默认的word-avg 模型进行训练；
                    attention-word-avg: 使用基于attention 的word-avg 模型进行训练；
                    self-attention-word-avg: 使用基于self-attention 的 word-avg 模型进行训练；
                    bert-classification: 使用基于bert的 分类模型进行训练】；

        :return:
        """
        if my_cuda_is_enable is False:
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.batch_size = batch_size
            self.epoch_size = epoch_size
            self.lr = lr
            self.dropout = dropout
            self.eval_steps = eval_steps
            self.lr_warmup = lr_warmup
            self.batch_split = batch_split
            self.weight_decay = weight_decay
            self.model_name = model_name
