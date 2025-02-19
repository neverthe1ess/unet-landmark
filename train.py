## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()


## 트레이닝 파라메터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue
# macOS apple slicon 수행 고려
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Resize((512,512)),Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    # 훈련만 셔플 사용
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    transform = transforms.Compose([Resize((512,512)),Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # best.pt를 저장하기 위한 평가 지표 설정
    best_val_dice = 0.0

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        # 지표 계산을 위해 누적할 리스트
        loss_arr = []
        iou_arr = []
        dice_arr = []
        precision_arr = []
        recall_arr = []
        f1_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            # 평가지표 계산
            loss_arr += [loss.item()]
            pred = (torch.sigmoid(output) > 0.5).float()
            iou, dice, precision, recall, f1 = get_segmentation_metrics(pred, label)
            iou_arr.append(iou)
            dice_arr.append(dice)
            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1)

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        # Epoch이 끝난 후 평균 Loss/지표 산출
        loss_epoch = np.mean(loss_arr)
        iou_epoch = np.mean(iou_arr)
        dice_epoch = np.mean(dice_arr)
        precision_epoch = np.mean(precision_arr)
        recall_epoch = np.mean(recall_arr)
        f1_epoch = np.mean(f1_arr)

        # Tensorboard 에 기록
        writer_train.add_scalar('loss', loss_epoch, epoch)
        writer_train.add_scalar('IoU', iou_epoch, epoch)
        writer_train.add_scalar('Dice', dice_epoch, epoch)
        writer_train.add_scalar('Precision', precision_epoch, epoch)
        writer_train.add_scalar('Recall', recall_epoch, epoch)
        writer_train.add_scalar('F1', f1_epoch, epoch)

        # ------------------
        # Validation
        # ------------------
        with torch.no_grad():
            net.eval()
            loss_arr = []

            # val 지표 계산 리스트
            iou_arr = []
            dice_arr = []
            precision_arr = []
            recall_arr = []
            f1_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                # 지표 계산
                pred = (torch.sigmoid(output) > 0.5).float()
                iou, dice, precision, recall, f1 = get_segmentation_metrics(pred, label)
                iou_arr.append(iou)
                dice_arr.append(dice)
                precision_arr.append(precision)
                recall_arr.append(recall)
                f1_arr.append(f1)

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        # Epoch 단위 평균
        loss_epoch = np.mean(loss_arr)
        iou_epoch = np.mean(iou_arr)
        dice_epoch = np.mean(dice_arr)
        precision_epoch = np.mean(precision_arr)
        recall_epoch = np.mean(recall_arr)
        f1_epoch = np.mean(f1_arr)

        writer_val.add_scalar('loss', loss_epoch, epoch)
        writer_val.add_scalar('IoU', iou_epoch, epoch)
        writer_val.add_scalar('Dice', dice_epoch, epoch)
        writer_val.add_scalar('Precision', precision_epoch, epoch)
        writer_val.add_scalar('Recall', recall_epoch, epoch)
        writer_val.add_scalar('F1', f1_epoch, epoch)

        # epoch 50 마다 저장
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        # ----- Best pt 모델 저장 -----
        if dice_epoch > best_val_dice:
            best_val_dice = dice_epoch
            print("***** Best performance updated - Saving the model *****")

            # best 모델로 저장
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, filename="model_best.pth")

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        iou_arr = []
        dice_arr = []
        precision_arr = []
        recall_arr = []
        f1_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            # 지표 계산
            pred = (torch.sigmoid(output) > 0.5).float()
            iou, dice, precision, recall, f1 = get_segmentation_metrics(pred, label)
            iou_arr.append(iou)
            dice_arr.append(dice)
            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1)


            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # 오버레이 이미지 저장
                fig, ax = plt.subplots(figsize=(5, 5))  # 개별 이미지 크기 조정
                ax.imshow(input[j].squeeze(), cmap='gray')  # 원본 input 표시
                ax.imshow(output[j].squeeze(), cmap='Reds', alpha=0.5)  # output을 빨간색으로 overlay
                ax.axis('off')  # 축 제거 (깔끔한 저장을 위해)
                plt.savefig(os.path.join(result_dir, 'overlay', 'overlay_%04d.png' % id), bbox_inches='tight', pad_inches=0)
                plt.close(fig)  # 메모리 절약을 위해 figure 닫기

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

        # 테스트 전체 평균 지표 출력
        iou_avg = np.mean(iou_arr)
        dice_avg = np.mean(dice_arr)
        precision_avg = np.mean(precision_arr)
        recall_avg = np.mean(recall_arr)
        f1_avg = np.mean(f1_arr)

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
    print(f"TEST METRICS :: IoU: {iou_avg:.4f}, Dice: {dice_avg:.4f}, "
              f"Precision: {precision_avg:.4f}, Recall: {recall_avg:.4f}, F1: {f1_avg:.4f}")
