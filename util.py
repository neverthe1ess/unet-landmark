import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch, filename=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if filename is None:
        filename = f"model_epoch{epoch}.pth"

    save_path = os.path.join(ckpt_dir, filename)

    # state_dict들만 묶어서 저장
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               save_path)

    print(f"Model saved at: {save_path}")

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## 평가 지표 계산 함수
def get_segmentation_metrics(pred, target, eps=1e-7):
    """
    pred, target: shape (N, 1, H, W) 가정.
    값은 0 또는 1로 이미 Thresholding이 끝난 상태여야 함.
    eps: 분모가 0이 되지 않도록 하는 작은 값.
    """
    # 배치 전체 픽셀에 대해 펼치기
    pred = pred.view(-1)
    target = target.view(-1)

    # TP, FP, FN 계산
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    # IoU
    iou = TP / (TP + FP + FN + eps)
    # Dice
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    # Precision
    precision = TP / (TP + FP + eps)
    # Recall
    recall = TP / (TP + FN + eps)
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return iou.item(), dice.item(), precision.item(), recall.item(), f1.item()