import torchinfo

from backbone import CspDarknet
from neck import PAN
from head import YoloHead
import torch.nn as nn


class YoloV5(nn.Module):
    def __init__(self, anchors, nc, model_type):
        super().__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul, wid_mul = depth_dict[model_type], width_dict[model_type]

        bc = int(wid_mul * 64)  # 64
        bd = max(round(dep_mul * 3), 1)  # 3
        self.backbone = CspDarknet(bc, bd)
        self.neck = PAN(bc, bd)
        self.head = YoloHead(bc, anchors, nc)

    def forward(self, x):
        return self.head(*self.neck(*self.backbone(x)))


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    nc = 1
    model_type = "s"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.rand(1, 3, 640, 640).to(device)
    model = YoloV5(anchors, nc, model_type).to(device)
    summary(model, input_size=(3, 640, 640))
    torchinfo.summary(model, input_size=(3, 640, 640), batch_dim=0)
    output = model(inputs)
    for o in output:
        print(o.shape)
