from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt as sqrt
from itertools import product as product
from detection import Detect

try:
    from typing import NoReturn, List, Literal
    IMPORT_COMPLETED = True
except ImportError:
    IMPORT_COMPLETED = False
    from typing import NoReturn, List


class SSD(nn.Module):
    if IMPORT_COMPLETED:
        def __init__(self, phase:Literal['train', 'test']='train', num_classes:int=21):
            super(SSD, self).__init__()
            self.phase = phase
            self.num_classes = num_classes
            self.vgg: nn.ModuleList = make_vgg()
            self.extras: nn.ModuleList = make_extras()
            self.L2Norm = L2Norm()
            self.loc: nn.ModuleList = make_loc(num_classes=num_classes)
            self.conf: nn.ModuleList = make_conf(num_classes=num_classes)
            dbox = PriorBox()
            self.priors = dbox.forward()
            if phase == 'test':
                self.detect = Detect()
    else:
        def __init__(self, phase: str='train', num_classes:int=21):
            super(SSD, self).__init__()
            self.phase = phase
            self.num_classes = num_classes
            self.vgg: nn.ModuleList = make_vgg()
            self.extras: nn.ModuleList = make_extras()
            self.L2Norm = L2Norm()
            self.loc: nn.ModuleList = make_loc(num_classes=num_classes)
            self.conf: nn.ModuleList = make_conf(num_classes=num_classes)
            dbox = PriorBox()
            self.priors = dbox.forward()
            if phase == 'test':
                self.detect = Detect()

    def forward(self, x: torch.Tensor):
        bs = len(x)
        out, lout, cout = list(), list(), list()
        for i in range(23):
            x = self.vgg[i](x)
        x1 = x
        out.append(self.L2Norm(x1))

        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        out.append(x)

        for i in range(0, 8, 2):
            x = F.relu(self.extras[i](x), inplace=True)
            x = F.relu(self.extras[i+1](x), inplace=True)
            out.append(x)
        
        for (x, l, c) in zip(out, self.loc, self.conf):
            lx = l(x).permute(0, 2, 3, 1).contiguous()
            cx = c(x).permute(0, 2, 3, 1).contiguous()
            lout.append(lx)
            cout.append(cx)
        lout = torch.cat([o.view(o.size(0), -1) for o in lout], 1)
        cout = torch.cat([o.view(o.size(0), -1) for o in cout], 1)

        lout = lout.view(lout.size(0), -1, 4)
        cout = cout.view(cout.size(0), -1, self.num_classes)

        output = (lout, cout, self.priors)

        if self.phase == 'test':
            return self.detect.apply(output, self.num_classes)
        else:
            return output


class L2Norm(nn.Module):
    def __init__(self, n_channels:int=512, scale:int=20) -> NoReturn:
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out: torch.Tensor = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

        return out

class PriorBox(object):
    def __init__(self) -> NoReturn:
        super(PriorBox, self).__init__()
        self.image_size: int = 300
        self.feature_maps: List[int] = [38, 19, 10, 5, 3, 1]
        self.steps: List[int] = [8, 16, 32, 64, 100, 300]
        self.min_size: List[int] = [30, 60, 111, 162, 213, 264]
        self.max_size: List[int] = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios: List[List[int]] = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self) -> torch.Tensor:
        mean: List[float] = list()
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + .5) / f_k
                cy = (i + .5) / f_k
                s_k = self.min_size[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_size[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output: torch.Tensor =  torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)

        return output