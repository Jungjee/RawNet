import math
from collections import OrderedDict

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data

from .modules import *
from .amsoftmax import *


class RawNet2(nn.Module):
    """
    Refactored RawNet2 architecture.

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    """

    def __init__(
        self,
        block,
        layers,
        nb_filters,
        nb_spk,
        code_dim=512,
        in_channels=1,
        margin=0.3,
        scale=15,
        **kwargs
    ):
        super(RawNet2, self).__init__()
        self.inplanes = nb_filters[0]

        #####
        # first layers before residual blocks
        #####
        self.conv1 = nn.Conv1d(
            in_channels, nb_filters[0], kernel_size=3, stride=3, padding=0
        )

        #####
        # residual blocks for frame-level representations
        #####
        self.layer1 = self._make_layer(block, nb_filters[0], layers[0])
        self.layer2 = self._make_layer(block, nb_filters[1], layers[1])
        self.layer3 = self._make_layer(block, nb_filters[2], layers[2])
        self.layer4 = self._make_layer(block, nb_filters[3], layers[3])
        self.layer5 = self._make_layer(block, nb_filters[4], layers[4])
        self.layer6 = self._make_layer(block, nb_filters[5], layers[5])

        #####
        # aggregate to utterance(segment)-level
        #####
        self.bn_before_agg = nn.BatchNorm1d(nb_filters[5])
        self.attention = nn.Sequential(
            nn.Conv1d(nb_filters[5], 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, nb_filters[5], kernel_size=1),
            nn.Softmax(dim=-1),
        )

        #####
        # speaker embedding layer
        #####
        self.fc = nn.Linear(nb_filters[5] * 2, code_dim)  # speaker embedding layer
        self.lrelu = nn.LeakyReLU(0.3)  # keras style

        #####
        # classification head
        #####
        self.class_head = LossFunction(
            nOut=code_dim, nClasses=nb_spk, margin=margin, scale=scale
        )

        #####
        # initialize
        #####
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.sig = nn.Sigmoid()

    def _make_layer(self, block, planes, nb_layer, downsample_all=False):
        if downsample_all:
            downsamples = [True] * (nb_layer)
        else:
            downsamples = [False] * (nb_layer - 1) + [True]
        layers = []
        for d in downsamples:
            layers.append(block(self.inplanes, planes, downsample=d))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x, is_test=False, label=None):
        # extract speaker embeddings.
        code = self.forward_encoder(x)

        if not is_test:
            loss = self.class_head(code, label)
            return loss

        else:
            return code

    def forward_encoder(self, x):
        #####
        # first layers before residual blocks
        #####
        x = self.conv1(x)

        #####
        # frame-level
        #####
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        #####
        # aggregation: attentive statistical pooling
        #####
        x = self.bn_before_agg(x)
        x = self.lrelu(x)
        w = self.attention(x)
        m = torch.sum(x * w, dim=-1)
        s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        x = torch.cat([m, s], dim=1)
        x = x.view(x.size(0), -1)

        #####
        # speaker embedding layer
        #####
        x = self.fc(x)

        return x


def get_RawNet2(layers, nb_filters, nb_spk, code_dim, block=RawNetBasicBlock, **kwargs):
    model = RawNet2(
        RawNetBasicBlock,
        layers=layers,
        nb_filters=nb_filters,
        nb_spk=nb_spk,
        code_dim=code_dim,
        **kwargs
    )
    return model


if __name__ == "__main__":
    from torchsummary import summary

    layers = [1, 1, 3, 4, 6, 3]
    nb_filters = [128, 128, 256, 256, 256, 256]
    nb_spk = 6112
    code_dim = 512
    model = RawNet2(
        RawNetBasicBlock,
        layers=layers,
        nb_filters=nb_filters,
        nb_spk=nb_spk,
        code_dim=code_dim,
    )
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (1, 59049), device=torch.device("cpu"))
