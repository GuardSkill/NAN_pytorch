import torch
from torch import nn
import torch.nn.functional as F
import os
from Inception import InceptionNet, BasicConv2d
from facenet_pytorch import MTCNN, InceptionResnetV1


def load_weights(model, model_path):
    """Download pretrained state_dict and load into model.

    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        model_path {str} -- Path of pretrained state_dict.

    """
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Load Pretrained weight successfully")
    else:
        print("Weight File not found")


class CNNNet(nn.Module):
    """特征抽取网络

    Arguments:
        feat_dim {int} -- Feacture vector dimension
        classify {bool} -- For classification task or verification task
        pretrained {str/None} -- The path of pretrained weights
        class_num {int} -- Class number (Person number)

    """

    def __init__(self, feat_dim=128, class_num=1595, classify=True, pretrained=None):
        super(CNNNet, self).__init__()
        self.cnn_model = InceptionNet(feat_size=128)
        self.fc = nn.Linear(feat_dim, class_num)
        self.classify = classify
        self.class_num = class_num
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        if pretrained is not None:
            load_weights(self, pretrained)

    def forward(self, img):
        feat = self.cnn_model(img)
        feat = F.normalize(feat, p=2, dim=1)
        if not self.classify:
            return feat
        else:
            logits = self.fc(feat)
            # logits = F.log_softmax(logits, dim=-1)
        return feat, logits

    def process(self, img, label):
        feat, logits = self(img)
        loss = nn.CrossEntropyLoss()(logits, label)  # softmax loss
        # One hot encoding buffer that you create out of the loop and just keep reusing
        _, index = logits.detach().max(dim=1)
        acc_num = index.eq(label).sum().item()
        return loss, acc_num

class Attention(nn.Module):
    """论文nan attention 模块"""

    def __init__(self, feat_dim=128):
        super(Attention, self).__init__()
        self.feat_dim = feat_dim
        self.q = nn.Parameter(torch.ones((1, 1, feat_dim)) * 0.0, requires_grad=True)
        # self.q = nn.Parameter(torch.ones((1, 1, feat_dim)), requires_grad=True)

        # self.q = nn.Parameter(torch.ones((1, 1, feat_dim)), requires_grad=False)

        self.fc = nn.Linear(feat_dim, feat_dim)
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(self.q)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0)
        self.fc.bias.data.zero_()  # NAN init these paprameter as zeros
        self.fc.weight.data.zero_()

    def squash(self, x):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        v = (x2 / (1.0 + x2)) * (x / x2.sqrt())
        return v

    def forward(self, Xs):
        # Xs: batch*feature-length*seq
        N, C, K = Xs.shape  # N: batch C: channel(feature dimention) K: Frames  [3,128,20]
        score = torch.matmul(self.q, Xs)  # N*1*K ==[1,1,128] *[3,128,20]
        score = F.softmax(score, dim=-1)
        r = torch.mul(Xs, score)  # element-wise multiply
        r = torch.sum(r, dim=-1)  # N*C

        new_q = self.fc(r)  # N*C
        new_q = self.tanh(new_q)
        new_q = new_q.view(N, 1, C)

        new_score = torch.matmul(new_q, Xs)
        new_score = F.softmax(new_score, dim=-1)

        o = torch.mul(Xs, new_score)
        o = torch.sum(o, dim=-1)  # N*C

        # o = self.squash(o)

        return o


class Online_Contrastive_Loss(nn.Module):
    def __init__(self, margin=2.0, num_classes=200):
        super(Online_Contrastive_Loss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, x, label):
        # compute pair wise distance
        n = x.size(0)
        xxt = torch.matmul(x, x.t())
        xn = torch.sum(torch.mul(x, x), keepdim=True, dim=-1)
        dist = xn.t() + xn - 2.0 * xxt

        one_hot_label = torch.zeros(x.size(0), self.num_classes)
        one_hot_label.scatter_(1, label.unsqueeze(-1), 1)
        pmask = torch.matmul(one_hot_label, one_hot_label.t())
        nmask = (1 - pmask)
        pmask[torch.eye(pmask.shape[0]) > 0] = 0.0

        pmask = pmask > 0
        nmask = nmask > 0

        ploss = torch.sum(torch.masked_select(dist, pmask)) / torch.sum(pmask)
        nloss = torch.sum(torch.clamp(self.margin - torch.masked_select(dist, nmask), min=0.0)) / torch.sum(nmask)
        #
        # mining the top k hardest negative pairs
        # neg_dist = torch.masked_select(-dist, nmask)
        # k = torch.sum(pmask)
        # neg_dist, _ = neg_dist.topk(k=k)
        # nloss = torch.sum(torch.clamp(self.margin + neg_dist, min=0.0))/k

        loss = (ploss + nloss)
        return loss


class Contrastive_Loss(nn.Module):
    def __init__(self, margin=2.0):
        super(Contrastive_Loss, self).__init__()
        self.margin = margin

    def __call__(self, l2, label):
        """
        :param l2: 
        :param label: Is the data pair are same person  1:represent same 0: not same
        :return: 
        """
        # one_tensor = torch.sum((r1) ** 2, dim=1)  debug
        loss_contrastive = torch.mean((label) * l2 +
                                      (1.0 - label) * torch.clamp(self.margin - l2, min=0.0))
        return loss_contrastive


class NANNet(nn.Module):
    """NAN 网络： 各个部件组合"""

    def __init__(self, feat_dim=128, ave_pool=False, cnn_path=None, fix_CNN=True):
        super().__init__()
        self.ave_pool = ave_pool
        self.CNN = CNNNet(classify=False, pretrained=cnn_path)
        self.attention = Attention(feat_dim=feat_dim)
        if fix_CNN:
            for param in self.CNN.parameters():
                param.requires_grad = False
        self.verification_loss = Contrastive_Loss()
        self.feat_dim = feat_dim
        self.fix_CNN=fix_CNN
        self.alpha = 1

    def forward(self, x1, x2):
        x1_frames = x1.view(x1.size(0) * x1.size(1), x1.size(2), x1.size(3), x1.size(4))
        feat_frames = self.CNN(x1_frames)
        r1 = feat_frames.view(x1.size(0), -1, x1.size(1))  # Batch 128 Frames
        # r1 = F.normalize(r1)
        # Aggregate the frames to one vector
        if self.ave_pool:
            r1 = torch.mean(r1, dim=-1)
        else:
            r1 = self.attention(r1)

        x2_frames = x2.view(x2.size(0) * x2.size(1), x2.size(2), x2.size(3), x2.size(4))
        feat_frames = self.CNN(x2_frames)
        r2 = feat_frames.view(x2.size(0), -1, x2.size(1))
        # r2 = F.normalize(r2)
        if self.ave_pool:
            r2 = torch.mean(r2, dim=-1)
        else:
            r2 = self.attention(r2)

        l2 = torch.sqrt(torch.sum((r1 - r2) ** 2, dim=1))
        return r1, r2, l2

    def process(self, x1, x2, label):
        _, _, l2 = self(x1, x2)
        loss = self.verification_loss(l2, label)
        return l2, loss

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.fix_CNN:
            self.CNN = self.CNN.eval()
        return self

class CNN_RGBDiff(nn.Module):
    """特征抽取网络

    Arguments:
        feat_dim {int} -- Feacture vector dimension
        classify {bool} -- For classification task or verification task
        pretrained {str/None} -- The path of pretrained weights
        class_num {int} -- Class number (Person number)

    """

    def __init__(self, feat_dim=128, class_num=1595, classify=True, pretrained=None):
        super(CNN_RGBDiff, self).__init__()
        self.cnn_model = InceptionNet(feat_size=128)
        self.cnn_model.conv1 = BasicConv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.fc = nn.Linear(feat_dim, class_num)
        self.classify = classify
        self.class_num = class_num
        if pretrained is not None:
            load_weights(self, pretrained)

    def forward(self, img):
        feat = self.cnn_model(img)
        feat = F.normalize(feat, p=2, dim=1)
        if not self.classify:
            return feat
        else:
            logits = self.fc(feat)
            # logits = F.log_softmax(logits, dim=-1)
        return feat, logits

    def process(self, img, label):
        feat, logits = self(img)
        loss = nn.CrossEntropyLoss()(logits, label)  # softmax loss
        # One hot encoding buffer that you create out of the loop and just keep reusing
        _, index = logits.detach().max(dim=1)
        acc_num = index.eq(label).sum().item()
        return loss, acc_num


class NANNet_RGBDiff(nn.Module):
    """NAN 网络： 各个部件组合"""

    def __init__(self, feat_dim=128, ave_pool=False, cnn_path='./checkpoints/cnn_RGBdiffer_model0.9958.pth', fix_CNN=True):
        super().__init__()
        self.ave_pool = ave_pool
        self.CNN = CNN_RGBDiff(classify=False, pretrained=cnn_path)
        self.attention = Attention(feat_dim=feat_dim)
        if fix_CNN:
            for param in self.CNN.parameters():
                param.requires_grad = False
        self.verification_loss = Contrastive_Loss()
        self.feat_dim = feat_dim
        self.fix_CNN=fix_CNN
        self.alpha = 1

    def forward(self, x1, x2):
        x1_frames = x1.view(x1.size(0) * x1.size(1), x1.size(2), x1.size(3), x1.size(4))
        feat_frames = self.CNN(x1_frames)
        r1 = feat_frames.view(x1.size(0), -1, x1.size(1))  # Batch 128 Frames
        # r1 = F.normalize(r1)
        # Aggregate the frames to one vector
        if self.ave_pool:
            r1 = torch.mean(r1, dim=-1)
        else:
            r1 = self.attention(r1)

        x2_frames = x2.view(x2.size(0) * x2.size(1), x2.size(2), x2.size(3), x2.size(4))
        feat_frames = self.CNN(x2_frames)
        r2 = feat_frames.view(x2.size(0), -1, x2.size(1))
        # r2 = F.normalize(r2)
        if self.ave_pool:
            r2 = torch.mean(r2, dim=-1)
        else:
            r2 = self.attention(r2)

        l2 = torch.sqrt(torch.sum((r1 - r2) ** 2, dim=1))
        return r1, r2, l2

    def process(self, x1, x2, label):
        _, _, l2 = self(x1, x2)
        loss = self.verification_loss(l2, label)
        return l2, loss

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.fix_CNN:
            self.CNN = self.CNN.eval()
        return self
