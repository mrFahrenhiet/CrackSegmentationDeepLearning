import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from averageMeter import AverageMeter


def init_log():
    log = {
        'loss': AverageMeter(),
        'time': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'acc': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'f1': AverageMeter()
    }
    return log


def compute_dice2(pred, gt):
    pred = (pred >= .5).float()
    dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
    return dice_score


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (BCE + dice_loss)

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice_loss


class log_cosh_dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(log_cosh_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        dice = compute_dice2(inputs, targets).item()
        loss = 1 - dice
        log_cosh = torch.log((torch.exp(loss) + torch.exp(-1 * loss)))
        return log_cosh


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


def get_IoU(outputs, labels):
    EPS = 1e-6
    outputs = (outputs > 0.5).int()
    labels = (labels > 0.5).int()
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0
    return iou.mean()


def accuracy(preds, label):
    preds = (preds > 0.5).int()
    label = (label > 0.5).int()
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc


def precision_recall_f1(preds, label):
    epsilon = 1e-7
    y_true = (label > 0.5).int()
    y_pred = (preds > 0.5).int()
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1


def confusion_mat(preds, label):
    y_true = (label > 0.5).int()
    y_pred = (preds > 0.5).int()
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    return tp, tn, fp, fn


if __name__ == '__main__':
    mask = torch.ones(1, 1, 128, 128)
    mask[:, :, 50:100, 50:100] = 0
    print("\n\n", mask[0][0], "\n\n")
    print("*************************************")
    print(mask.shape)
    print(compute_dice2(mask, mask))
    print(compute_dice2(mask, torch.zeros(mask.shape)))

    loss = TverskyLoss()
    print(loss(mask, mask), loss(mask, torch.zeros(mask.shape)))

    print(get_IoU(mask, mask), get_IoU(mask, torch.zeros(mask.shape)))

    print(accuracy(mask, torch.zeros(mask.shape)))

