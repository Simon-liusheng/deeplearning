import warnings
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    # //如果是空洞卷积
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        '''
        nn.conv2d函数的基本参数是：
        nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, paddinng_mode='zeros')
        参数：nn.Conv 参考链接https://blog.csdn.net/qq_26369907/article/details/88366147
        c1:输入数据的通道数，例RGB图片的通道数为3
        c2:输出数据的通道数，这个根据模型调整
        kennel_size:卷积核大小，可以是int，或tuple，kennel_size=2,意味着卷积大小（2,2）， kennel_size=(2,3),意味着卷积大小（2,3），即非正方形卷积
        stride: 步长，默认为1， 与kennel_size类似， stride=2, 意味着步长上下左右扫描皆为2， stride=(2,3),左右三秒步长为2，上下为3
        padding: 零填充
        groups: 从输入通道到输出通道阻塞连接数,通道分组的参数，输入通道数，输出通道数必须同时满足被groups整除
        groups：如果输出通道为6，输入通道也为6，假设groups为3，卷积核为1*1,；则卷积核的shape为2*1*1，即把输入通道分成了3份；那么卷积核的个数呢？之前是由输出通道
            决定的，这里也一样，输出通道为6，那么就有6个卷积核。这里实际是将卷积核也平分为groups份，在groups份特征图上计算，以输入输出都为6为例，每个2*h*w的特征图子层
            就有且仅有2个卷积核，最后相加恰好是6，这里可以起到的作用是不同通道分别计算特征
        bias：如果为True，则向输出添加可学习的偏置
        '''
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else act if isinstance(act, nn.Module) else nn.Identity()

    # 前向计算,cba
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # 前向融合计算
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, groups=1, e=0.5):
        super().__init__()
        # hidden channels
        _c = int(c2 * e)
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(_c, c2, 3, 1, groups=groups)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, groups=1, e=0.5):
        super().__init__()
        # hidden channels
        _c = int(c2 * e)
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(c1, _c, 1, 1)
        self.cv3 = Conv(2 * _c, c2, 1)
        self.bottleneck_n = nn.Sequential(*(Bottleneck(_c, _c, shortcut, groups, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(
            torch.cat(
                tensors=(self.cv1(x),
                         self.bottleneck_n(self.cv2(x))),
                dim=1)
        )


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):  # 多次k=5的pooling等同于spp的不同kernel size的pooling
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            return self.cv2(torch.cat((x, y1, y2, y3), 1))
