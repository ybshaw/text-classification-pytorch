# _*_ coding:utf-8 _*_
# @Time     :2022/3/31 19:35
# @Author   :ybxiao 
# @FileName :train_eval.py
# @Software :PyCharm


import numpy as np
from tqdm import trange
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertTokenizer
from utils import EarlyStopping, CommonModelDataset, BertModelDataset
from utils import get_dataloader, read_labels, read_corpus, get_vocab, get_label2idx, set_random_seed
from models import Bert, TextCNN, TextRNN, TextRCNN, TextRNNAttention, TextDPCNN


def train(train_loader, dev_loader, model, epoch, loss_func, device=None, is_bert_model=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-2)
    early_stopping = EarlyStopping(patience=2, verbose=False)

    for i in trange(epoch, desc='Epoch'):
        print("Training Epoch:[{}/{}]".format(i + 1, epoch))
        tr_loss, tr_acc = 0, 0
        n_steps, n_examples = 0, 0
        model.train()

        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)                   # (batch, seq_len)
            print(input_ids.size(), label_ids.size(), attn_mask.size())

            label_ids = label_ids.squeeze()                             # (batch, )
            if is_bert_model:
                output = model(input_ids, attn_mask=attn_mask)          # (batch, n_labels)
            else:
                output = model(input_ids)
            loss = loss_func(output, label_ids)
            tr_loss += loss.item()
            n_steps += 1

            if idx % 100 == 0:
                avg_loss = tr_loss / n_steps
                print('batch loss is:{:.5f}'.format(avg_loss))

            pred = torch.argmax(output, axis=1)                         # (batch, )
            tmp_acc = accuracy_score(label_ids.cpu().numpy(), pred.cpu().numpy())
            tr_acc += tmp_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = tr_loss / n_steps
        tr_acc = tr_acc / n_steps
        print("Training loss:{:.5f} \t Training acc:{:.4f}".format(epoch_loss, tr_acc))

        dev_loss, dev_acc = valid(dev_loader, model, loss_func, device, is_bert_model)
        print('dev loss: {:.5f} \t dev acc:{:.4f}'.format(dev_loss, dev_acc))
        early_stopping(dev_loss, model)
        if early_stopping.early_stop:
            early_stopping.save_checkpoint(dev_loss, model)
            break


def valid(dev_loader, model, loss_func, device=None, is_bert_model=False):
    dev_loss, dev_acc = 0, 0
    n_steps = 0
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)                   # (batch, seq_len)

            label_ids = label_ids.squeeze()                             # (batch, )
            if is_bert_model:
                output = model(input_ids, attn_mask=attn_mask)          # (batch, n_labels)
            else:
                output = model(input_ids)
            loss = loss_func(output, label_ids)
            dev_loss += loss.item()
            n_steps += 1

            pred = torch.argmax(output, axis=1)                         # (batch, )
            tmp_acc = accuracy_score(label_ids.cpu().numpy(), pred.cpu().numpy())
            dev_acc += tmp_acc
    dev_loss = dev_loss / n_steps
    dev_acc = dev_acc / n_steps

    return dev_loss, dev_acc


def evaluate(test_loader, model, label2idx, device=None, save_model_path=None, is_bert_model=False):
    preds, targets, = [], []
    idx2label = {v:k for k,v in label2idx.items()}
    if save_model_path:
        model.load_state_dict(torch.load(save_model_path))
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)

            if is_bert_model:
                output = model(input_ids, attn_mask=attn_mask)      # (batch, n_labels)
            else:
                output = model(input_ids)

            output = torch.argmax(output, axis=1)                   # (batch, )
            label_ids = label_ids.squeeze()                         # (batch, )

            preds.extend([idx2label[idx.item()] for idx in output])
            targets.extend([idx2label[idx.item()] for idx in label_ids])
    report = classification_report(y_true=targets, y_pred=preds, digits=4)
    return report



if __name__ == "__main__":
    set_random_seed(seed=2022)

    label_path = 'dataset/TNEWS/labels.json'
    train_corpus_path = 'dataset/TNEWS/train.json'
    dev_corpus_path = 'dataset/TNEWS/dev.json'

    labels_list =read_labels(label_path)
    train_corpus = read_corpus(train_corpus_path)      # train: 53360
    vocab2idx = get_vocab(train_corpus)
    label2idx = get_label2idx(labels_list)

    dev_data = read_corpus(dev_corpus_path)           # len: 10000
    np.random.shuffle(dev_data)
    dev_corpus = dev_data[:8000]                      # dev: 8000
    test_corpus = dev_data[8000: ]                    # test: 2000

    train_params = {"batch_size":32, "shuffle":True}
    dev_params = {"batch_size":32, "shuffle":False}
    test_params = {"batch_size": 64, "shuffle": False}

    bert_model_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    train_loader = get_dataloader(train_corpus, vocab2idx, label2idx, max_len=30, params=train_params)
    dev_loader = get_dataloader(dev_corpus, vocab2idx, label2idx, max_len=30, params=dev_params)
    test_loader = get_dataloader(test_corpus, vocab2idx, label2idx, max_len=30, params=test_params)

    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextCNN(filter_size=[2, 3, 4], n_filters=100, vocab_size=len(vocab2idx), embed_size=200,
                    seq_len=20, dropout=0.3, n_labels=len(label2idx)).to(device)
    # model = Bert(bert_model_path, labels_list).to(device)

    train(train_loader, dev_loader, model, epoch=1, loss_func=loss_func, device=device, is_bert_model=True)

    res = evaluate(test_loader, model, label2idx, device, save_model_path='checkpoint.pt', is_bert_model=True)
    print(res)