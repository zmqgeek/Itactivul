# -*- coding: utf-8 -*-

import random
import re
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn import metrics
from print_log import Logger
from gensim.models import FastText
import sys
import itertools
from tqdm import tqdm

def mean(z): # used for FastText and Word2Vec
    return sum(itertools.chain(z))/len(z)


# save training log
logdir = 'log/'
sys.stdout = Logger(logdir + "thunderbird.log")


PAD = '<PAD>'
# sentence max length
pad_size = 80

tokenizer = lambda x: x.split(' ')  # word-level

# loading FastText word embedding
fasttext_model = FastText.load('archi_fasttext.model')

# dataset file path
file_path = 'data/thunderbird_report.txt'

all_data = []
data_label = []
# load dataset
with open(file_path, 'r', encoding='UTF-8') as f:
    for line in tqdm(f, desc='load data.....'):
        lin = line.strip()
        if not lin:
            continue
        # get dataset text and label
        content,label = lin.split('\t')
        
        data_label.append(label)
        # tokenizer
        
        token = tokenizer(content.lower())
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        
        result = [fasttext_model.wv[w.lower()] for w in token]
    
        feature = [mean(x) for x in zip(*result)]
        all_data.append(feature)
  


X_train, X_test, Y_train, Y_test = train_test_split(all_data, data_label,test_size=0.5,random_state=random.randint(1,20191112))

#L=['rf','nb','knn','svm','gbdt','lr']
#L=['rf','knn','svm','gbdt','lr']
L=['lr']


result=[]
for n in L:
    if n=='rf':
        clf = RandomForestClassifier()
    elif n=='nb':
        clf = GaussianNB()
    elif n=='knn':
        clf = KNeighborsClassifier()
    elif n=='svm':
        clf = svm.SVC()
    elif n=='mlp':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    elif n=='lr':
        clf = LogisticRegression()

    
    print('training......')
    clf.fit(X_train, Y_train)
    predict_all = clf.predict(X_test)
    
    
    print('testing......')
    test_report = metrics.classification_report(Y_test, predict_all,digits=4)
    confusion = metrics.confusion_matrix(Y_test, predict_all)
    
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


