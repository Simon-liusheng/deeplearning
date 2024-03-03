
import torch.nn as nn


class YoloHead(nn.Module):
    def __init__(self, bc, anchors, nc):
        super().__init__()
        self.no = nc + 5  # number of outputs per anchor
        self.na = len(anchors[0]) // 2  # number of anchors
        self.p3_head = nn.Conv2d(bc * 4, self.no * self.na, kernel_size=1)
        self.p4_head = nn.Conv2d(bc * 8, self.no * self.na, kernel_size=1)
        self.p5_head = nn.Conv2d(bc * 16, self.no * self.na, kernel_size=1)

    def forward(self, p3, p4, p5):
        p3 = self.p3_head(p3)
        p4 = self.p4_head(p4)
        p5 = self.p5_head(p5)
        return p3, p4, p5
