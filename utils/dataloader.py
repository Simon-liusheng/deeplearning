import os
import random

import cv2
import math
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.transform import Albumentations, mixup, letterbox, augment_hsv, copy_paste

"""
dataloader 主要继承于torch得Dataset，重写关键的三个魔法函数
init: 需要整理好数据文件地址，包括图片地址以及标签地址
len: 返回数据集大小
★★★★★ getitem: 根据传入的index，返回经过预处理图片数据、标签，以及原始图片相关信息
"""


class YoloDataset(Dataset):
    def __init__(self, mode, hyp, transform=None):
        self.mode = mode
        self.hyp = hyp
        self.img_size = hyp.size
        self.img_paths, self.label_paths = self.get_filenames()
        self.labels = self.load_labels()
        # 设计transform模块，需要__call__()方法
        self.transform = transform
        n = len(self.label_paths)
        self.indices = np.arange(n)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        # opencv读取的图片数值为BGR格式，PIL库读取图片得到的RGB格式

        hyp = self.hyp
        if self.mode == "train":
            # 保证随机
            if random.random() < hyp["mosaic"]:
                img, labels = self.load_mosaic(index)
                if random.random() < hyp["mixup"]:
                    img, labels = mixup(img, labels, *self.load_mosaic(random.choice(self.indices)))
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            # Letterbox
            img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

    def get_filenames(self):
        path = self.hyp[self.mode]
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        with open(path) as f:
            img_paths = [x.strip('\n') for x in f.readlines()]
        label_paths = [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]
        return img_paths, label_paths

    def load_image(self, index):
        # 读取图片，预处理到到 640 尺度，但是不填充
        im = cv2.imread(self.img_paths[index])
        h0, w0 = im.shape[:2]
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]

    def load_label(self):
        pass

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4


def yolo_collate_fn(self, batch):
    pass


def create_dataloader(mode, hyp):
    dataset = YoloDataset(mode, hyp)
