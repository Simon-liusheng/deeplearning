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
    def __init__(self, opt):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def train_transforms(self):
        pass

    def val_transforms(self):
        pass

    def test_transforms(self):
        pass


def yolo_collate_fn(self, batch):
    pass

