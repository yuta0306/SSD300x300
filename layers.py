import torch
import torch.nn as nn

from typing import List, Literal, NoReturn
from torch.nn.modules import conv, padding

from torch.nn.modules.pooling import MaxPool2d


def make_vgg() -> nn.ModuleList:
    cfg: List[Literal['M', 'C'] or int] = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C',
            512, 512, 512, 'M', 512, 512, 512]
    in_channels = 3
    layers: List = list()
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                        nn.ReLU(inplace=False)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True),
                conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def make_extras() -> nn.ModuleList:
    layers: List[nn.Conv2d] = [
        nn.Conv2d(1024, 256, kernel_size=(1)),
        nn.Conv2d(256, 512, kernel_size=(3), stride=2, padding=1),
        nn.Conv2d(512, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3), stride=2, padding=1),
        nn.Conv2d(256, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3)),
        nn.Conv2d(256, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3))
    ]

    return nn.ModuleList(layers)

def make_loc(num_classes=21) -> nn.ModuleList:
    layers: List[nn.Conv2d] = [
        # Out1
        nn.Conv2d(512, 4*4, kernel_size=3, padding=1),
        # Out2
        nn.Conv2d(1024, 6*4, kernel_size=3, padding=1),
        # Out3
        nn.Conv2d(512, 6*4, kernel_size=3, padding=1),
        # Out4
        nn.Conv2d(256, 6*4, kernel_size=3, padding=1),
        # Out5
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),
        # Out6
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1)
    ]

    return nn.ModuleList(layers)

def make_conf(num_classes=21) -> nn.ModuleList:
    layers: List[nn.Conv2d] = [
        # Out1
        nn.Conv2d(512, 4*num_classes, kernel_size=3, padding=1),
        # Out2
        nn.Conv2d(1024, 6*num_classes, kernel_size=3, padding=1),
        # Out3
        nn.Conv2d(512, 6*num_classes, kernel_size=3, padding=1),
        # Out4
        nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1),
        # Out5
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1),
        # Out6
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1),
    ]

    return nn.ModuleList(layers)

if __name__ == "__main__":
    print(make_vgg())
    print(make_extras())
    print(make_loc())
    print(make_conf(12))