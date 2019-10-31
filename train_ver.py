#!/usr/bin/env python
# coding: utf-8
import os
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm
from Dataset import YTBDatasetVer, YTBDatasetCNN, YTBDatasetVer_RGBdiff
from Network import NANNet, CNNNet, NANNet_RGBDiff
import numpy as np
# In[9]:
from util import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 设置跑第几个GPU


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join("./", figure_name), dpi=fig.dpi)


def train_NAN():
    # 使用cuda运算
    device = torch.device("cuda")
    dataset = YTBDatasetVer(csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=224)
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=16, num_workers=2)

    # ### 初始化萌新（模型）
    model = NANNet(cnn_path='./cnn_model6.pth').to(device)

    model = model.train()

    # ### 查看可以更新的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # ### 读取存储好的NAN模型权值
    # model.load_state_dict(torch.load("nan_model.pth"))
    # # 训练
    # model.init_weights()
    acc_max = 0
    optimizer = torch.optim.Adadelta(model.parameters(), lr=3)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    # optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0005, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    for epoch in range(300):
        total_loss = 0
        total_size = 0
        bar = tqdm(dataload)
        labels, distances = [], []
        for i, (x1, x2, y) in enumerate(bar):
            optimizer.zero_grad()
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            l2, loss = model.process(x1, x2, y)
            total_size += x1.size(0)
            loss.backward()
            optimizer.step()
            # b=pred.item()
            distances.append(l2.detach().data.cpu().numpy())
            labels.append(y.cpu().numpy())
            total_loss += loss.item()

            bar.set_postfix(loss=f"{total_loss/(i+1):0.4f}",
                            epoch=f"{epoch+1}")
        labels = np.concatenate(labels)
        distances = np.concatenate(distances)

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
        plot_roc(fpr, tpr, figure_name="roc_train_epoch_{}.png".format(epoch))

        acc = np.mean(accuracy)
        torch.save(model.state_dict(), "nan_model.pth")
        if acc_max < acc:
            acc_max = max(acc, acc_max)
            torch.save(model.state_dict(), f"./checkpoints/nan_model_acc{acc_max:0.4f}.pth")
        if acc > 0.8:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1 - acc)
        if acc > 0.8:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1 - acc)


def train_NAN_RGBDiff():
    # 使用cuda运算
    device = torch.device("cuda")
    dataset = YTBDatasetVer_RGBdiff(csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=224,
                                    num_frames=100)
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=8
                                           , num_workers=4)

    # ### 初始化萌新（模型）
    model = NANNet_RGBDiff(cnn_path='./checkpoints/cnn_RGBdiffer_model0.9958.pth').to(device)

    model = model.train()
    model.load_state_dict(torch.load("./checkpoints/nan_RGB_diff_acc0.8149.pth"))
    # ### 查看可以更新的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # ### 读取存储好的NAN模型权值
    # model.load_state_dict(torch.load("nan_model.pth"))
    # # 训练
    # model.init_weights()
    acc_max = 0
    # optimizer = torch.optim.Adadelta(model.parameters(),lr=1e-3)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)

    # optimizer = torch.optim.Adadelta(model.parameters(), lr=3)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-7)  #record
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    # optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0005, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    for epoch in range(300):
        total_loss = 0
        total_size = 0
        bar = tqdm(dataload)
        labels, distances = [], []
        for i, (x1, x2, y) in enumerate(bar):
            optimizer.zero_grad()
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            l2, loss = model.process(x1, x2, y)
            total_size += x1.size(0)
            loss.backward()
            optimizer.step()
            # b=pred.item()
            distances.append(l2.detach().data.cpu().numpy())
            labels.append(y.cpu().numpy())
            total_loss += loss.item()

            bar.set_postfix(loss=f"{total_loss/(i+1):0.4f}",
                            epoch=f"{epoch+1}")
        labels = np.concatenate(labels)
        distances = np.concatenate(distances)

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
        plot_roc(fpr, tpr, figure_name="roc_RGB_diff_epoch_{}.png".format(epoch))

        acc = np.mean(accuracy)
        torch.save(model.state_dict(), "nan_RGB_diff.pth")
        if acc_max < acc:
            acc_max = max(acc, acc_max)
            torch.save(model.state_dict(), f"./checkpoints/nan_RGB_diff_acc{acc_max:0.4f}.pth")
        if acc > 0.8:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=1 - acc)
        if acc > 0.8:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=1 - acc)


if __name__ == '__main__':
    train_NAN_RGBDiff()
