import os
import numpy as np

import cv2

import torch
import torch.nn as nn

from PIL import Image
from numpy import dtype


## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label_path = os.path.join(self.data_dir, self.lst_label[index])
        input_path = os.path.join(self.data_dir, self.lst_input[index])

        label_pil = Image.open(label_path).convert("L")
        label = np.array(label_pil, dtype=np.uint8)

        input_pil = Image.open(input_path).convert("L")
        input = np.array(input_pil, dtype=np.uint8)

        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
        
class Resize(object):
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, data):
        label, input_ = data['label'], data['input']

        # label이나 input_이 (H, W, C) 형태라고 가정
        # label이 (H, W, 1)이면 그대로 resize 가능
        # cv2의 dsize=(width, height)이므로 self.size가 (w, h)여야 함
        # 라벨은 NEAREST, input은 AREA 보간
        label_resized = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        input_resized = cv2.resize(input_, dsize=self.size, interpolation=cv2.INTER_AREA)
        
        #label이 2D가 되어버렸다면 (H, W, 1)로 확장
        if label_resized.ndim == 2:
            label_resized = label_resized[:, :, np.newaxis]

        #만약 input이 흑백이라면(드문 경우) 여기도 필요
        if input_resized.ndim == 2:
            input_resized = input_resized[:, :, np.newaxis]
        
        data['label'] = label_resized
        data['input'] = input_resized    
            
        return data        

