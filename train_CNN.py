import os
import torch
from tqdm import tqdm
from Dataset import YTBDatasetCNN, YTBDatasetCNN_RGBDiff
from Network import CNNNet, CNN_RGBDiff

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 设置跑第几个GPU


def train_CNN():
    # 使用cuda运算
    device = torch.device("cuda")
    batch_size = 16

    dataset = YTBDatasetCNN(csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=224)
    num_class = int(len(dataset) / dataset.person_frames)
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    len(dataset)
    model = CNNNet(class_num=num_class).to(device)
    model = model.train()
    # model.load_state_dict(torch.load("model.acc.pth"))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    acc_max = 0
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=3)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

    # label_one_hot = torch.LongTensor(batch_size,  num_class).to(device)
    # label_one_hot.zero_()
    for epoch in range(300):
        total_loss = 0
        total_acc = 0
        total_size = 0
        bar = tqdm(dataload)
        for i, (x, y, _) in enumerate(bar):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            # label=y.unsqueeze(1)
            # In your for loop
            # label_one_hot.scatter_(1, label, 1)
            loss, acc_num = model.process(x, y)
            total_size += x.size(0)
            loss.backward()
            optimizer.step()
            # b=pred.item()
            total_acc += acc_num
            total_loss += loss.item()

            bar.set_postfix(loss=f"{total_loss/(i+1):0.4f}",
                            acc=f"{total_acc/total_size:0.4f}",
                            epoch=f"{epoch+1}")
        acc = total_acc / total_size
        torch.save(model.state_dict(), "cnn_model.pth")
        if acc_max < acc:
            acc_max = max(total_acc / total_size, acc_max)
            torch.save(model.state_dict(), f"./checkpoints/cnn_model_acc{acc_max:0.4f}.pth")


def train_CNN_RGBDiff():
    # 使用cuda运算
    device = torch.device("cuda")
    batch_size = 16
    dataset = YTBDatasetCNN_RGBDiff(csv_file='../splits.txt', root_dir='../aligned_images_DB', img_size=224)
    num_class = int(len(dataset) / dataset.person_frames)
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    len(dataset)
    model = CNN_RGBDiff(class_num=num_class).to(device)
    model = model.train()

    # model.load_state_dict(torch.load("cnn_RGBdiffer_model.pth"))
    model.load_state_dict(torch.load("./checkpoints/cnn_RGBdiffer_model0.9958.pth"))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    acc_max = 0
    optimizer = torch.optim.RMSprop(model.parameters(),  lr=5e-4)


    # label_one_hot = torch.LongTensor(batch_size,  num_class).to(device)
    # label_one_hot.zero_()
    for epoch in range(300):
        total_loss = 0
        total_acc = 0
        total_size = 0
        bar = tqdm(dataload)
        for i, (x, d, y) in enumerate(bar):
            optimizer.zero_grad()
            x, d, y = x.to(device), d.to(device), y.to(device)

            x = torch.cat([x, d], dim=1)
            # label=y.unsqueeze(1)
            # In your for loop
            # label_one_hot.scatter_(1, label, 1)
            loss, acc_num = model.process(x, y)
            total_size += x.size(0)
            loss.backward()
            optimizer.step()
            # b=pred.item()
            total_acc += acc_num
            total_loss += loss.item()

            bar.set_postfix(loss=f"{total_loss/(i+1):0.4f}",
                            acc=f"{total_acc/total_size:0.4f}",
                            epoch=f"{epoch+1}")
        acc = total_acc / total_size
        torch.save(model.state_dict(), "cnn_RGBdiffer_model.pth")
        if acc_max < acc:
            acc_max = max(total_acc / total_size, acc_max)
            torch.save(model.state_dict(), f"./checkpoints/cnn_RGBdiffer_model{acc_max:0.4f}.pth")
if __name__=='__main__':
    train_CNN_RGBDiff()