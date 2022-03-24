# _*_ coding:utf-8 _*_
# @Time     :2022/3/23 19:47
# @Author   :ybxiao 
# @FileName :models.py
# @Software :PyCharm

"""分类任务中常用的经典模型"""



import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTModel(nn.Module):
    def __init__(self, model_path, labels_list):
        super(BertModel, self).__init__()
        self.model_name = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(model_path)
        self.bert_config = BertConfig.from_pretrained(model_path)
        self.hidden_size = self.bert_config.hidden_size
        self.n_labels = len(labels_list)
        self.linear = nn.Linear(self.hidden_size, self.n_labels)

    def forward(self):


