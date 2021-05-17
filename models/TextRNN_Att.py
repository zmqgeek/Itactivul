# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """config par setting"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN_Att'
        self.train_path = dataset + '/data/train.txt'                                # train dataset
        self.dev_path = dataset + '/data/dev.txt'                                    # dev dataset
        self.test_path = dataset + '/data/chromium_report.txt'                                  # test dataset
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # class name
        self.vocab_path = dataset + '/data/vocab.pkl'                                # vocab dict
        self.save_path = dataset + '/saved_dict/' + self.model_name +'.ckpt'        # saved model
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained embedding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # device: cpu or gpu
        self.dropout = 0.5                                              
        self.require_improvement = 200                                 
        self.num_classes = len(self.class_list)                         # class num
        self.n_vocab = 0                                                # vocab size
        self.num_epochs = 20                                            # epoch_num
        self.batch_size = 2                                           # mini-batch大小
        self.pad_size = 80                                              # sentence size
        self.learning_rate = 1e-3                                       # learning rate
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # word embedding
        self.hidden_size = 128                                          # lstm hidden layer
        self.num_layers = 2                                             # lstm layer_num
        self.hidden_size2 = 64


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        #feature = emb
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        #feature = out
        out = F.relu(out)
        out = self.fc1(out)
        #feature = out
        out = self.fc(out)  # [128, 64]
        return out
