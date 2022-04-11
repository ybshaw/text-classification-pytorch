# text-classification-pytorch

基于PyTorch的一个简单中文文本分类项目demo，目的是为了将常用的分类模型走一遍流程，并在必要的地方给出了详细的注释，涉及到的模型有:textCNN、textRNN、textRCNN、textRNN_attn、BERT预训练模型。

本次项目在实现过程中也参考了网络上很多优秀作者的开源项目，文末会尽量给出引用出处，如果有未提供引用的还望告知，笔者会加上去

### 数据集

因为主要是为了熟悉常用的分类模型，所以数据集采用的是`THEWS`数据集(今日头条中文新闻（短文本）)，[数据集详细说明和下载地址](https://www.cluebenchmarks.com/introduce.html)

数据集示例：

> {"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}

主要包括15种类别，**并且由于原始数据集中的测试集并没有提供对应标签用于验证模型性能，因此笔者从验证集中随机选取了20%一共2000条样本作为测试集验证**

### 环境
依赖环境保存在`requirements.txt`文件中，执行命令`pip install -r requirements.txt`即可安装

### 模型

尝试了常见的几种经典模型，主要如下:
- TextCNN
- TextRNN
- TextRCNN
- TextRNN + attention
- BERT

### 超参数

- embed_size = 200
- seq_len = 30 (短补长截)
- epoch = 20
- lr = 3e-3(BERT为5e-5)

> 实验过程中还采用了提前停止：如果在验证集上的loss超过3个epoch还没有下降时，则提前结束训练

### 结果

实验中采用的数据集规模：

- train: 53360
- dev: 8000
- test: 2000

| 模型              | precision（%） | recall（%） | F1（%） |
| :---------------- | -------------- | ----------- | ------- |
| TextCNN           | 45.97         | 49.32         | 48.93   |
| TextRNN           | 42.66 | 42.31 | 41.76 |
| TextRCNN          | 46.69 | 45.39 | 45.37 |
| TextRNN_attention | 44.41 | 43.56 | 43.68 |
| DPCNN             | 47.39 | 43.64 | 44.35 |
| BERT              | 55.51 | 54.19 | 54.52 |
> 模型中的BERT模型采用的版本是`bert-base-chinese`,[下载地址](https://huggingface.co/bert-base-chinese/tree/main)，将其中的`vocab.txt`，`config.json`以及`pytorch_model.bin`三个文件放到项目目录下的`bert-base-chinese`下即可

> 不得不说，预训练模型就是好用，啥也没干直接在目标数据集上微调下，仅仅训练两个epoch就已经达到54.52%的F1值了，已经比其它方法超出至少六七个百分点

### 总结

根据结果可以看到，几种模型相比较于目前的[SOTA结果](https://www.cluebenchmarks.com/classification.html)(目前基于预训练模型的SOTA大约在70+，普通模型的结果暂时没看到一般是多少）还有挺大差距，当然也因为本项目的目的只是为了进一步熟悉这些常用的分类模型，所以在参数上没有细调以及优化，当然有兴趣的也可以有针对性的做进一步优化，简单罗列了几种：

1. 目前采用的只是简单的交叉熵损失函数，实际上可以采用鲁棒性更强的损失函数如**Focal Loss**
2. 虽然训练集的规模为5w+也已经不少了，但对于模型而言一般还是远远不够，可以尝试采用一些**数据增强**策略来增强样本，或者加入**对抗训练**来避免过拟合
3. 使用预训练好的词向量来作为embedding，例如该[项目提供的词向量](https://github.com/Embedding/Chinese-Word-Vectors)
4. 可以采用**半监督学习**的方式来获得更多的训练数据
5. 通过**主动学习**来获得质量更高的样本
6. 模型层采用其它的初始化方式如**Xavier**或者**Kaiming**等方式来替代默认的初始化方式
7. 以BERT为代表的预训练模型可以采用性能更好的模型如：RoBerta等
8. BERT的输出层可以采用最后一层的均值池化来代替`[CLS]`作为句向量，目前大多数论文的结果均表明采用均值池化的输出效果会好一点(仅仅是个人理解，没有理论依据支撑)

### 引用

- TextCNN:  [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

- TextRNN: [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/pdf/1605.05101.pdf)
- TextRCNN: [Recurrent Convolutional Neural Network for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9745)
- TextRNN_attn: [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034.pdf)
- DPCNN: [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://aclanthology.org/P17-1052.pdf)
- BERT: [BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [关于DPCNN模型结构的解释](https://zhuanlan.zhihu.com/p/35457093)

- [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)（大佬写的非常简洁易懂，感谢大佬的开源）