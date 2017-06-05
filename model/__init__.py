#
# Created by Aman LaChapelle on 5/17/17.
#
# pytorch-feedbacknet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-feedbacknet/LICENSE.txt
#

from .ConvLSTM import ConvLSTM, ConvLSTMCell


import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
