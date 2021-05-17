# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    
    # 加载数据集
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # 读取每一行
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            # 以\t切分得到数据内容 [1]为label
            content = lin.split('\t')[0]
            # tokenizer为切分方式，对于每一行切分的字
            for word in tokenizer(content):
                # get()返回字典里指点键的值，如果不存在则为默认值
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 排序词表，取出词表限制长度的词
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 添加UNK,PAD到字典中
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        # vocab为字典序, {word: id}
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
        
    # 如果存在词表，则加载词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    # 否则创建词表，将其写入词表文件
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_ner_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, content_ner, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                # 得到词典对应的词id，如果词不存在则为默认值UNK的ID
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # ner id
                # token = tokenizer(content_ner)
                # seq_len = len(token)
                # ner_pad_size = 80
                # if ner_pad_size:
                #     if len(token) < ner_pad_size:
                #        token.extend([PAD] * (ner_pad_size - len(token)))
                #    else:
                #        token = token[:ner_pad_size]
                #        seq_len = ner_pad_size
                # word to id
                #for word in token:
                # 得到词典对应的词id，如果词不存在则为默认值UNK的ID
                #    words_line.append(vocab.get(word, vocab.get(UNK)))
                # 加入词的id列表，label以及长度
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                # 得到词典对应的词id，如果词不存在则为默认值UNK的ID
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # 加入词的id列表，label以及长度
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    # 加载数据集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    # 加载ner数据集
    #train = load_ner_dataset(config.train_path, config.pad_size)
    #dev = load_ner_dataset(config.dev_path, config.pad_size)
    #test = load_ner_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    
    # 得到词汇表
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    
    # 随机初始化每个词的词嵌入向量
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    # 加载预训练词向量
    f = open(pretrain_dir, "r", encoding='UTF-8')
    # 读取词向量的每一行
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        # 如果词向量的词在字典里
        if lin[0] in word_to_id:
            # 得到词对应的词典id
            idx = word_to_id[lin[0]]
            # 替换300维的预训练词向量，并转换为float类型
            emb = [float(x) for x in lin[1:301]]
            # 将列表转换为float32数组
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    # 将初始化后的词向量存储
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
