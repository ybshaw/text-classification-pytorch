# _*_ coding:utf-8 _*_
# @Time     :2022/3/23 18:50
# @Author   :ybxiao 
# @FileName :utils.py
# @Software :PyCharm

"""
数据的预处理
"""

import random
import numpy as np
import json
import torch
import jieba
from torch.utils.data import Dataset, DataLoader


# 提前停止
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=2, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CommonModelDataset(Dataset):
    """
    非预训练模型的数据预处理:
    因为常规模型在预处理的时候需要分词，所以会有点不同
    """
    def __init__(self, corpus, vocab2idx, label2idx, max_len=None):
        self.corpus = corpus
        self.max_len = max_len
        self.vocab2idx = vocab2idx
        self.label2idx = label2idx

    def __getitem__(self, item):
        sentence, label, _ = self.corpus[item]
        word_list = jieba.lcut(sentence)
        input_id = [self.vocab2idx.get(word, 1) for word in word_list]
        label_id = [self.label2idx.get(label)]

        if len(input_id) < self.max_len:
            input_id.extend([0] * (self.max_len - len(input_id)))
        else:
            input_id = input_id[:self.max_len]
        # 这里取名为attn_mask其实不太合适，因为模型没有使用到attention，所以一般直接叫mask,
        # 之所以叫attn_mask，仅仅只是为了和BERT模型中保持一致，方便后续调用，减少不必要的条件判断
        attn_mask = [1 if idx != 0 else 0 for idx in input_id]

        return {
            "input_ids": torch.tensor(input_id, dtype=torch.long),
            "label_ids": torch.tensor(label_id, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self.corpus)


class BertModelDataset(Dataset):
    """预训练模型的预处理，这里主要就是BERT"""
    def __init__(self, corpus, tokenizer, max_len, label2idx):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(corpus)
        self.label2idx = label2idx

    def __getitem__(self, item):
        sentence, label, _ = self.corpus[item]
        tokenized = self.tokenizer.encode_plus(sentence,
                                               add_special_tokens=True,
                                               max_length=self.max_len,
                                               padding='max_length',
                                               return_attention_mask=True,
                                               truncation=True)
        input_ids = tokenized['input_ids']
        attn_mask = tokenized['attention_mask']
        label_ids = [self.label2idx[label]]

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'label_ids': torch.tensor(label_ids, dtype=torch.long),
                'attn_mask': torch.tensor(attn_mask, dtype=torch.long)
                }

    def __len__(self):
        return self.len

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

def get_vocab(corpus):
    vocab = {"<pad>": 0, "<unk>":1}
    for item in corpus:
        sentence = item[0]
        word_list = jieba.lcut(sentence)
        for word in word_list:
            if not word in vocab:
                vocab[word] = len(vocab)
    return vocab

def get_label2idx(labels_list):
    labe2idx = {label:idx for idx, label in enumerate(labels_list)}
    return labe2idx

def get_dataloader(corpus, vocab2idx, label2idx, tokenizer=None, is_bert_model=False, max_len=None, params=None):
    if is_bert_model:
        dataset = BertModelDataset(corpus, tokenizer, max_len, label2idx)
    else:
        dataset = CommonModelDataset(corpus, vocab2idx, label2idx, max_len)
    dataloader = DataLoader(dataset, **params)
    return dataloader

def set_random_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    pass
