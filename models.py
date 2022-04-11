# _*_ coding:utf-8 _*_
# @Time     :2022/3/23 19:47
# @Author   :ybxiao 
# @FileName :models.py
# @Software :PyCharm

"""分类任务中常用的经典模型"""



import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertConfig


class Bert(nn.Module):
    def __init__(self, model_path, labels_list):
        super(Bert, self).__init__()
        self.model_name = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(model_path)
        self.bert_config = BertConfig.from_pretrained(model_path)
        self.hidden_size = self.bert_config.hidden_size             # bert hidden_size: 768
        self.n_labels = len(labels_list)
        self.linear = nn.Linear(self.hidden_size, self.n_labels)

    def forward(self, input_ids=None, attn_mask=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)
        sequence_out, pool_out = bert_out[0], bert_out[1]
        output = self.linear(pool_out)                              # (batch, n_labels)
        return output


class TextCNN(nn.Module):
    def __init__(self, filter_size, n_filters, vocab_size, embed_size, seq_len=None, dropout=None, n_labels=None):
        super(TextCNN, self).__init__()
        self.model_name = 'text_cnn'
        self.filter_size= filter_size
        self.n_filters = n_filters
        self.n_labels = n_labels

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_size, n_filters, kernel_size=k),
                          nn.BatchNorm1d(n_filters),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=seq_len-k+1))
            for k in filter_size])
        self.linear = nn.Linear(n_filters * len(filter_size), self.n_labels)

        # 自定义embedding层的初始化方式（optional）
        # self.embedding.weight.data.copy_(torch.from_numpy(self.init_embedding(
        #     vocab_size, embed_size)))

    def init_embedding(self, vocab_size, embed_size):
        """
        自定义embedding层参数初始化（非必须）来替代默认的随机初始化
        效果有时候会好一点，需要视不同情况而定
        :param vocab_size:
        :param embed_size:
        :return:
        """
        init_embedding = np.empty([vocab_size, embed_size])
        scale = np.sqrt(3.0 / embed_size)
        for idx in range(vocab_size):
            init_embedding[idx:] = np.random.uniform(-scale, scale, [embed_size])
        return init_embedding

    def forward(self, x):
        embed = self.embedding(x)               # (batch,seq_len,embed_size)
        # 一维卷积是在最后一个维度上卷积，所以需要把seq_len放到最后一个维度
        embed = embed.permute(0, 2, 1)          # (batch, embed_size, seq_len)
        out = [conv(embed) for conv in self.convs]
        out = torch.cat(out, dim=1)             # (batch,num_filter * len(filter_size),1)
        out = out.view(-1, out.size(1))
        out = self.linear(self.dropout(out))    # (batch, n_labels)
        return out


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=None, n_labels=None):
        super(TextRNN, self).__init__()
        self.model_name = 'text_rnn'
        if hidden_size % 2 != 0:
            raise ValueError('LSTM的维度:{}不是偶数，需要设置为偶数'.format(hidden_size))
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size, n_labels)

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)          # (batch, seq_len, hidden_size)
        output = lstm_out[:, -1, :]             # 取最后时刻作为输出: (batch, hidden_size)
        output = self.linear(output)            # (batch, n_labels)
        return output


class TextRCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len=None, dropout=None, n_labels=None):
        super(TextRCNN, self).__init__()
        self.model_name = 'text_rcnn'
        if hidden_size % 2 != 0:
            raise ValueError('输入的LSTM的维度:{} 不是偶数，需要设置为偶数'.format(hidden_size))
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(embed_size + hidden_size, n_labels)
        self.max_pool = nn.MaxPool1d(kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.embedding(x)                       # (batch, seq_len, embed_size)
        lstm_out, _ = self.lstm(embed)                  # (batch, seq_len, hidden_size)
        # 在最后一个维度拼接
        output = torch.cat((lstm_out, embed), dim=-1)   # (batch, seq_len, hidden_size+embed_size)

        # 一维池化是在最后一个维度上进行，所以需要把seq_len放到最后一个维度上
        output = self.max_pool(output.permute(0, 2, 1)) # (batch, hidden_size+embed_size, 1)
        output = output.squeeze()                       # (batch, hidden_size+embed_size)
        output = self.linear(self.dropout(output))      # (batch, n_labels)

        return output


class TextRNNAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=None, n_labels=None):
        super(TextRNNAttention, self).__init__()
        self.model_name = 'text_rnn_attn'
        if hidden_size % 2 != 0:
            raise ValueError('LSTM的维度:{}不是偶数，需要设置为偶数'.format(hidden_size))
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.w = nn.Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(hidden_size, n_labels)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)           # (batch, seq_len, embed_size)
        lstm_out, _ = self.lstm(embed)      # (batch, seq_len, hidden_size)

        alpha = self.softmax(torch.matmul(lstm_out, self.w)).unsqueeze(-1)  # (batch, seq_len, 1)
        # 以上步骤可以表述为如下三步:
        # 1. 矩阵相乘: matmul( (batch, seq_len, hidden_size), (hidden_size, ) ) = (batch, seq_len)
        # 2. 在seq_len的维度上用softmax转换为概率: (batch, seq_len)
        # 3. 拓展维度: (batch, seq_len, 1)

        output = lstm_out * alpha          # (batch, seq_len, hidden_size)
        output = torch.sum(output, dim=1)  # (batch, hidden_size)
        output = self.linear(output)       # (batch, n_labels)

        return output


class TextDPCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_size=3, seq_len=None, dropout=None, n_labels=None):
        super(TextDPCNN, self).__init__()
        self.model_name = 'text_dpcnn'
        self.n_filters = n_filters
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.pad_0 = nn.ZeroPad2d((0, 0, 0, 1))  # 在最下面PAD一层0
        self.pad_1 = nn.ZeroPad2d((0, 0, 1, 1))  # 在上面和下面都PAD一层0

        # 为了避免使用ZeroPad使用时的麻烦，将以下的卷积池化操作全部换成Conv2d (使用Conv1d也可以，但需要转换下相对更麻烦)
        self.max_pool = nn.MaxPool2d(kernel_size=(filter_size, 1), stride=2)
        # 输入输出维度不一致的情况
        self.conv_region = nn.Conv2d(1, n_filters, kernel_size=(filter_size, embed_size), stride=1)
        # 输出输出的维度一致的情况
        self.conv = nn.Conv2d(n_filters, n_filters, kernel_size=(filter_size, 1), stride=1)
        self.linear = nn.Linear(2 * n_filters, n_labels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def _block(self, x):
        """repeat block:
        1/2 pooling + conv + conv
        """
        # 每经过一次block，seq_len的长度都变为原来的一半
        # 因为这里用的池化的步长为2，所以每次池化后seq_len都减半

        # 1/2池化,需要先pad一层0
        x = self.pad_0(x)               # 仅在下面PAD
        pool_x = self.max_pool(x)

        # 第一个等长卷积
        x = self.pad_1(pool_x)          # 上下都PAD
        x = self.relu(x)
        x = self.conv(x)

        # 第二个等长卷积
        x = self.pad_1(x)
        x = self.relu(x)
        x = self.conv(x)

        # 残差连接
        x = x + pool_x
        return x

    def forward(self, x):
        x = self.embedding(x)           # (batch, seq_len, embed_size)
        x = x.unsqueeze(1)              # (batch, 1, seq_len,embed_size)
        x = self.conv_region(x)         # (batch, num_filters, seq_len-3+1, 1)

        x = self.pad_1(x)               # 上下PAD: (batch, num_filters, seq_len, 1)
        x = self.relu(x)
        x = self.conv(x)                # (batch, num_filters, seq_len-3+1, 1)

        x = self.pad_1(x)               # (batch, num_filters, seq_len, 1)
        x = self.relu(x)
        x = self.conv(x)                # (batch, num_filters, seq_len-3+1, 1)

        while x.size()[2] > 2:          # seq_len > 2时,x.size()[2] = seq_len
            x = self._block(x)          # 由于1/2池化层的存在，seq_len会随着block的增加而指数减少

        x = x.squeeze().view(-1, 2 * self.n_filters)
        output = self.linear(x)         # (batch, n_labels)

        return output


if __name__ == "__main__":
    pass