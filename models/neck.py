import torch
import torch.nn as nn
from common import Conv, C3


class YoloNeck(nn.Module):
    def __init__(self, bc, bd):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_p5 = Conv(bc * 16, bc * 8, 1, 1)
        self.conv_p4 = Conv(bc * 8, bc * 4, 1, 1)
        self.c3_p4 = C3(bc * 16, bc * 8, bd, shortcut=False)
        self.c3_p3 = C3(bc * 8, bc * 4, bd, shortcut=False)
        self.conv_downsample_p3 = Conv(bc * 4, bc * 4, kernel_size=3, stride=2)
        self.conv_downsample_p4 = Conv(bc * 8, bc * 8, kernel_size=3, stride=2)
        self.p4_c3 = C3(bc * 8, bc * 8, bd, shortcut=False)
        self.p5_c3 = C3(bc * 16, bc * 16, bd, shortcut=False)

    def forward(self, p3, p4, p5):
        p5 = self.conv_p5(p5)
        p5_upsample = self.upsample(p5)
        p4 = self.conv_p4(self.c3_p4(torch.cat(tensors=(p4, p5_upsample), dim=1)))
        p4_upsample = self.upsample(p4)
        p3 = self.c3_p3(torch.cat(tensors=(p3, p4_upsample), dim=1))
        p3_downsample = self.conv_downsample_p3(p3)
        p4 = self.p4_c3(torch.cat(tensors=(p3_downsample, p4), dim=1))
        p4_downsample = self.conv_downsample_p4(p4)
        p5 = self.p5_c3(torch.cat(tensors=(p4_downsample, p5), dim=1))
        return p3, p4, p5
