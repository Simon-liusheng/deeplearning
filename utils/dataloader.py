import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# opencv读取的图片数值为BGR格式，PIL库读取图片得到的RGB格式

#
class YoloDataset(Dataset):
    def __init__(self, path, transform=None):
        self.img_paths, self.label_paths = self.get_filenames(path)
        # 设计transform模块，需要__call__()方法
        self.transform = transform

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).getdata())
        with open(self.label_paths[idx]) as f:
            # 训练的bbox格式为x,y,w,h
            gt_bbox = [x.strip('\n').split('/')[-1] for x in f.readlines()]
        sample = {'img': image, 'gt_bbox': gt_bbox}
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def get_filenames(path):
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        with open(path) as f:
            img_paths = [x.strip('\n') for x in f.readlines()]
        label_paths = [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]
        return img_paths, label_paths


def yolo_collate_fn(self, batch):
    pass
