# _*_ coding:utf-8 _*_
# @Time     :2022/3/23 18:50
# @Author   :ybxiao 
# @FileName :utils.py
# @Software :PyCharm


import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


def read_labels(file):
    labels_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            label = data['label_desc']
            labels_list.append(label)
    return labels_list

def read_corpus(file):
    corpus = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            label, sentence, keywords = data['label_desc'], data['sentence'], data['keywords']
            corpus.append((sentence, label, keywords))
    return corpus


class Data(Dataset):
    def __init__(self, corpus, tokenizer, max_len, labels_list):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(corpus)
        self.label2idx = {label: idx for idx, label in enumerate(labels_list)}

    def __getitem__(self, item):
        sentence, label, _ = self.corpus[item]
        tokenized = self.tokenizer.encode_plus(sentence,
                                               add_special_tokens=True,
                                               max_length=self.max_len,
                                               padding='max_length',
                                               return_attention_mask=True,
                                               truncation=True)
        input_ids = tokenized['input_ids']
        attn_mask = tokenizer['attention_mask']
        label_ids = self.label2idx['label']

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'label_ids': torch.tensor(label_ids, dtype=torch.long),
                'attn_mask': torch.tensor(attn_mask, dtype=torch.long)
                }

    def __len__(self):
        return self.len


if __name__ == "__main__":
    # label_path = 'dataset/TNEWS/labels.json'
    # labels = read_labels(label_path)
    # print(labels, len(labels))  # 15

    # corpus_path = 'dataset/TNEWS/dev.json'
    # corpus = read_corpus(corpus_path)
    # print(corpus[0])

    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # res = tokenizer.encode_plus(text='我有一直铅笔盒一个apple',
    #                       add_special_tokens=True,
    #                       max_length=15,
    #                       padding='max_length',
    #                       return_attention_mask=True,
    #                       truncation=True)
    # print(res)
    # print(tokenizer.convert_ids_to_tokens(res['input_ids']))

    from transformers import BertConfig
    config = BertConfig.from_pretrained('bert-base-chinese')
    print(config.hidden_size)

