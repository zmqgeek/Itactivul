# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import get_time_dif
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num=12, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor(w_2).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) 
        class_mask = F.one_hot(target, 12) 
        ids = target.view(-1, 1) 

        alpha_list = target.cpu().numpy().tolist()

        alpha_dict = {}
        for i in range(12):
            alpha_dict[i] = alpha_dict.get(i, 1)

        for i,label in enumerate(alpha_list):
            alpha_dict[label] = alpha_dict.get(i, 1) + 1
        
        count = []
        middle_list = []

        for value in alpha_dict.values():
            middle_list.append(value)
        middle_list.sort()
        middle = middle_list[10]
        middle = (middle_list[5] + middle_list[6]) // 2
        for key in sorted(alpha_dict):
            count.append(1)
            #count.append(middle/alpha_dict[key])
        
        alpha = torch.tensor(count).cuda()

        alpha = alpha[ids.data.view(-1)]
        # alpha = self.alpha[ids.data.view(-1)] 
        probs = (pt * class_mask).sum(1).view(-1, 1) 
        log_p = probs.log()
        # weight loss
        #loss = -(1-alpha) * log_p
        # focal loss
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

# init weight，default is xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # batch number
    dev_best_loss = float('inf')
    last_improve = 0  
    flag = False  # improve ture or false
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() 
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()

            #loss = F.cross_entropy(outputs, labels)
            focalloss = MultiCEFocalLoss()
            loss = focalloss(outputs, labels)
            #print('training loss')
            #print(loss)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], 
                 c=plt.cm.Set1(label[i] / 12.), marker = 'o')
    # c=y, cmap='Set1'
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    #return fig
    
 
def build_sentence_vector(sentence):
    sen_vec=np.zeros(sentence.shape[1]).reshape((1,sentence.shape[1]))
    for i in range(sentence.shape[0]):
        sen_vec = sen_vec + sentence[i]
    sen_vec = sen_vec/sentence.shape[0]
    return sen_vec

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, features, labels, weight_all = evaluate(config, model, test_iter, test=True)
    print(features.shape)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    print("Write sentence word weight to file...")
    all_weight_data_str = '\n'.join(weight_all)
    f = open('sentence_pattern_weight.txt', 'w', encoding='utf-8')
    f.write(all_weight_data_str)
    f.close()

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    print('T-SNE feature...')
    result = tsne.fit_transform(features)
    print("Starting Drawing Features Classification...")
    pdf = PdfPages('ohsumed_gcn_doc_test_2nd_layer.pdf')
    plt.scatter(result[:,0], result[:,1], c=labels, cmap='Set1')
    
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    pdf.close()
    # fig = plot_embedding(result, labels,
                        # 't-SNE embedding of the digits (time %.2fs)'
                        # % (time.time() - t0))

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    weight_all = []
    # feature matrix
    features = np.empty(shape=[0,64],dtype='float32')
    #features = np.empty(shape=[0,256],dtype='float32')
    with torch.no_grad():
        for texts, labels in tqdm(data_iter,desc='testing......'):
            outputs, feature, alpha = model(texts)
            # print(alpha.cpu().numpy()[0])
            # print(alpha.size())
            #print(outputs.size())
            feature = feature.cpu().numpy()
            #print(feature.shape)
            #print(feature[0].shape)
            
            for i in range(feature.shape[0]):
                features = np.append(features, feature[i].reshape(1,64), axis=0)

            focalloss = MultiCEFocalLoss()
            loss = focalloss(outputs, labels)
            #loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            
            weight_label = labels.tolist()
            weight_sentence = alpha.cpu().numpy()
            for i in range(np.shape(weight_sentence)[0]):
                weight_str = ' '.join(map(str,weight_sentence[i].ravel().tolist()))
                weight_str = weight_str + '\t' + str(weight_label[i])
                weight_all.append(weight_str)

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, features, labels_all, weight_all
    return acc, loss_total / len(data_iter)