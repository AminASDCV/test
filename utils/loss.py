import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce)

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        loss = self.nll_loss(log_p, targets)
        return loss


def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight * weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (self.weights * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

def dice_loss(y_true, y_pred, smooth = 1e-6):
    pred = y_pred.contiguous()
    target = y_true.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    pred = y_pred.contiguous()
    target = y_true.contiguous()

    numerator = 2 * torch.sum(torch.mul(pred,target))

    denominator = torch.sum(torch.pow(pred,2)) + torch.sum(torch.pow(pred,2))

    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

def calc_loss(pred, target, weight_bce):
    pred = pred
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(y_true = target, y_pred = pred)

    loss = bce * weight_bce + dice * (1-weight_bce)

    return loss
