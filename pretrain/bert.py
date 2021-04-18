from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, BertTokenizer
from transformers import BertConfig
from utils.common_util import (load_vocab)

# bert_config = BertConfig.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=bert_config)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# sequence = """
#     We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
#      from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
#       representations from unlabeled text by jointly conditioning on both left and right context in all layers.
#       As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create
#       state-of-the-art models for a wide range of tasks, such as question answering and language inference,
#       without substantial task-specific architecture modifications.
# """
# token = tokenizer(sequence)
# print(tokenizer.pad_token_id)
# '''
#     输出结构为    字典 {
#                         input_ids:'[]' 表示文本
#                         token_type_ids:'', 表示文本中的句子
#                         attention_mask:''  表示
#                     }
# '''
# print(token)
res, a, b = load_vocab("../data/label_vocab.txt")
print(res)
print(a)
print(b)
