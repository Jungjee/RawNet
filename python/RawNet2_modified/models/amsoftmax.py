#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)
# from: https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/loss/amsoftmax.py
# Adapted for PyTorch mixed precision and DDP training

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss().cuda(torch.cuda.current_device())
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label=None):

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda(torch.cuda.current_device())
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        return loss