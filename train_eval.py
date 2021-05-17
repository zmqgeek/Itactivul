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


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num=12, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        # 按照数据比例的权重
        w_1 = [0.0019753809009931777, 0.006090757778062297, 0.04210487809339162, 0.00018290563898084978, 0.0287344758838915, 0.37045708119181314, 0.0005852980447387193, 0.527335247745688, 0.0019205092092989227, 0.006237082289246978, 0.0013900828562544583, 0.012986300367640334]
        # 数据除以中位数
        w_2 = [0.2147117296222664, 0.6620278330019881, 4.576540755467197, 0.019880715705765408, 3.1232604373757455, 40.26640159045726, 0.0636182902584493, 57.31809145129225, 0.20874751491053678, 0.6779324055666004, 0.15109343936381708, 1.411530815109344]
        #数据除以中位数取倒数
        w_3 = [4.657407407407407, 1.5105105105105106, 0.21850564726324934, 50.3, 0.32017823042647997, 0.024834600572726375, 15.71875, 0.017446498560577155, 4.79047619047619, 1.4750733137829912, 6.618421052631579, 0.7084507042253522]
        if alpha is None:
            self.alpha = torch.tensor(w_2).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, 12) #获取target的one hot编码
        ids = target.view(-1, 1) 

        alpha_list = target.cpu().numpy().tolist()
        # 初始化标签字典
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
        # alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        # weight loss
        #loss = -(1-alpha) * log_p
        # focal loss
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class focal_loss(nn.Module):
    def __init__(self, alpha=[0.0019753809009931777, 0.006090757778062297, 0.04210487809339162, 0.00018290563898084978, 0.0287344758838915, 0.37045708119181314, 0.0005852980447387193, 0.527335247745688, 0.0019205092092989227, 0.006237082289246978, 0.0013900828562544583, 0.012986300367640334], gamma=2, num_classes=12, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            #print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            #print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds_softmax, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        print(alpha)
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        print(loss)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

# 权重初始化，默认xavier
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

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()

            loss = F.cross_entropy(outputs, labels)
            #focalloss = MultiCEFocalLoss()
            #loss = focalloss(outputs, labels)
            #print('training loss')
            #print(loss)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
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
                # 验证集loss超过1000batch没下降，结束训练
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
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    #return fig

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, features, labels, weight_all = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    #print("Write sentence word weight to file...")
    #all_weight_data_str = '\n'.join(weight_all)
    #f = open('Sentence_test_weight.txt', 'w', encoding='utf-8')
    #f.write(all_weight_data_str)
    #f.close()

    # 降维可视化
    #tsne = TSNE(n_components=2, init='pca', random_state=0)
    #t0 = time.time()
    #print("Starting Drawing Features Classification...")
    #result = tsne.fit_transform(features)
    #fig = plot_embedding(result, labels,
    #                     't-SNE embedding of the digits (time %.2fs)'
    #                     % (time.time() - t0))

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    weight_all = []
    # 定义一个空的特征数组
    features = np.empty(shape=[0,12],dtype='float32')
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # print(alpha.cpu().numpy()[0])
            # print(alpha.size())
            # 将输出的所有特征添加到数组
            features = np.append(features,outputs.cpu().numpy(),axis=0)

            #focalloss = MultiCEFocalLoss()
            #loss = focalloss(outputs, labels)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            
            weight_label = labels.tolist()
            #weight_sentence = alpha.cpu().numpy()
            #for i in range(np.shape(weight_sentence)[0]):
            #    weight_str = ' '.join(map(str,weight_sentence[i].ravel().tolist()))
            #    weight_str = weight_str + '\t' + str(weight_label[i])
            #    weight_all.append(weight_str)

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, features, labels_all, weight_all
    return acc, loss_total / len(data_iter)