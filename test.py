import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Dataset import YouTubeBeDataset
from Network import NANnet

if __name__ == '__main__':
    # ----------------- test training -------------------
    batch_size = 8
    total_acc = 0
    total_size = 0
    device = torch.device("cuda")
    model = NANnet(device=device, threshold=0.5)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # if os.path.exists("model.acc.pth"):
    #     model.load_state_dict(torch.load("model.acc.pth"))
    # print(model)
    model = model.train()
    acc_max = 0
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.0005)
    dataset = YouTubeBeDataset()
    dataload = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    for epoch in range(300):
        total_loss = 0
        total_acc = 0
        total_size = 0
        for i, (x1, x2, y) in enumerate(dataload):
            optimizer.zero_grad()
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            r1, r2, acc, loss = model.process(x1, x2, y)
            total_size += x1.size(0)*x1.size(1)

            loss.backward()
            optimizer.step()
            # b=pred.item()
            total_acc += acc
            total_loss += loss.item()

            print('loss',total_loss/(i+1),'acc',total_acc/total_size,'epoch',epoch+1)
        acc = total_acc / total_size
        torch.save(model.state_dict(), "model.acc.pth")
        if acc_max < acc:
            acc_max = max(total_acc / total_size, acc_max)
            torch.save(model.state_dict(), f"./checkpoints/model.acc{acc_max:0.4f}.pth")

    # ----------------- test split.ext file  -------------------
    # split_facedata = pd.read_csv('/home/lulu/Dataset/splits.txt')
    # print(split_facedata.iloc[0, 3])
    # sub_df = split_facedata[split_facedata['split number'] == 2]
    # name = sub_df[' first name'].iloc[0]  # get city code
    # print(name)
