# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, test
#from test_data import test
from importlib import import_module
from print_log import Logger
import argparse
import sys

# required=True
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='TextRNN_Att', help='choose a model: TextRNN, TextRNN_Att')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--data', type=str, default='test.txt' help='choose a test dataset: chromium_report.txt, php_report.txt, thunderbird.txt')
parser.add_argument('--run', type=str, default='train',help='choose run way: train, test')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'tactical_word2vec_data'  # dataset

    # embedding
    embedding = 'embedding_word2vec_archi.npz'
    if args.embedding == 'random':  
        embedding = 'random'
    model_name = args.model  # TextRNN_Att
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
 
    # import_module() 
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    
    # save train log
    logdir = config.log_path + '/'
    sys.stdout = Logger(logdir + "{}.log".format(args.model))

    # set train random seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...")
    
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    # running time
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab) # vocab size
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    
    if args.run == 'train':  
        train(config, model, train_iter, dev_iter, test_iter)
    elif args.run == 'test':
        test(config, model, test_iter)
