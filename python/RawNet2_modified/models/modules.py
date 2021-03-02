import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data


class RawNetBasicBlock(nn.Module):
    """
    Basic block for RawNet architectures.
    This block follows pre-activation[1].

    Arguments:
    downsample  : perform reduction in the sequential(time) domain
                  (different with shortcut)

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    """

    def __init__(self, inplanes, planes, downsample=None):
        super(RawNetBasicBlock, self).__init__()
        self.downsample = downsample

        #####
        # core layers
        #####
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mp = nn.MaxPool1d(3)
        self.afms = AFMS(planes)
        self.lrelu = nn.LeakyReLU(0.3)

        #####
        # settings
        #####
        if inplanes != planes:  # if change in number of filters
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        out = self.lrelu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.lrelu(self.bn2(out)))
        out = out + shortcut

        if self.downsample:
            out = self.mp(out)
        out = self.afms(out)

        return out


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim):
        super(AFMS, self).__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x
