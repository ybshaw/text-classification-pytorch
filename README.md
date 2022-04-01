# text-classification-pytorch

基于PyTorch的一个简单中文文本分类项目demo，目的是为了将常用的分类模型走一遍流程，并在必要的地方给出了详细的注释，涉及到的模型有:textCNN、textRNN、textRCNN、textRNN_attn、BERT预训练模型。
本次项目在实现过程中也参考了网络上很多优秀作者的开源项目，文末会尽量给出引用出处，如果有未提供引用的，可以提个issue，笔者会加上去
---
### 数据集
因为主要是为了熟悉常用的分类模型，所以数据集采用的是`THEWS`数据集(今日头条中文新闻（短文本）)，[数据集详细说明和下载地址](https://www.cluebenchmarks.com/introduce.html)
主要包括15种类别，并且由于原始数据集中的测试集并没有提供对应标签用于验证模型性能，因此笔者从验证集中随机选取了20%一共2000条样本作为测试集验证

### 模型
尝试了常见的几种经典模型，主要如下:
- TextCNN
- TextRNN
- TextRCNN
- TextRNN + attention
- BERT

### 超参数

- embed_size = 200
- seq_len = 20 (短补长截)
- epoch = 50
- lr = 1e-3

### 结果

| 模型              | precision | recall | F1   |
| ----------------- | --------- | ------ | ---- |
| TextCNN           |           |        |      |
| TextRNN           |           |        |      |
| TextRCNN          |           |        |      |
| TextRNN_attention |           |        |      |
| DPCNN             |           |        |      |
| BERT              |           |        |      |



### 引用