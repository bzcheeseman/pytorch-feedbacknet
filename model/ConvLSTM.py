#
# Created by Aman LaChapelle on 5/17/17.
#
# pytorch-feedbacknet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-feedbacknet/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable

from typing import Union, Callable, List
import math

# See http://feedbacknet.stanford.edu/feedback_networks_2016.pdf for paper


class ConvLSTMCell(nn.Module):  # Stack-2, Stack-3 would have an extra Conv/BN
    def __init__(self,
                 input_filters: int,
                 hidden_filters: int,
                 kernel: Union[int, tuple]=3,
                 stride: Union[int, tuple]=1,
                 padding: Union[int, tuple]=0):
        super(ConvLSTMCell, self).__init__()

        self.hidden_filters = hidden_filters

        self._cuda = False

        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.Wxi = nn.Sequential(
            nn.Conv2d(input_filters, hidden_filters, kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel/2)),  # don't reduce size now
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Whi = nn.Sequential(
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel/2)),  # stride on this too?
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Wxf = nn.Sequential(
            nn.Conv2d(input_filters, hidden_filters, kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Whf = nn.Sequential(
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel/2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Wxc = nn.Sequential(
            nn.Conv2d(input_filters, hidden_filters, kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Whc = nn.Sequential(
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel/2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Wxo = nn.Sequential(
            nn.Conv2d(input_filters, hidden_filters, kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

        self.Who = nn.Sequential(
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel/2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU(),
            nn.Conv2d(hidden_filters, hidden_filters, kernel, padding=math.floor(kernel / 2)),
            nn.BatchNorm2d(hidden_filters),
            nn.ReLU()
        )

    def cuda(self, device_id=None):
        super(ConvLSTMCell, self).cuda(device_id)
        self._cuda = True

    def init_hidden(self, batch_size, input_dims):
        h_size = (
            math.floor((input_dims[0] - self.kernel[0] + 2*self.padding[0])/self.stride[0] + 1),
            math.floor((input_dims[1] - self.kernel[1] + 2*self.padding[1])/self.stride[1] + 1)
        )
        Hd0 = Variable(torch.zeros(batch_size, self.hidden_filters, int(h_size[0]), int(h_size[1])))
        Cd0 = Variable(torch.zeros(batch_size, self.hidden_filters, int(h_size[0]), int(h_size[1])))

        if self._cuda:
            return Hd0.cuda(), Cd0.cuda(), h_size
        else:
            return Hd0, Cd0, h_size

    def forward(self, Xdm1t, Hdtm1, Cdtm1, Hdtmn=None):
        idt = Funct.sigmoid(self.Wxi(Xdm1t) + self.Whi(Hdtm1))
        fdt = Funct.sigmoid(self.Wxf(Xdm1t) + self.Whf(Hdtm1))

        Ctilde = Funct.tanh(self.Wxc(Xdm1t) + self.Whc(Hdtm1))
        Cdt = fdt * Cdtm1 + idt * Ctilde

        odt = Funct.sigmoid(self.Wxo(Xdm1t) + self.Who(Hdtm1))
        Hdt = odt * Funct.tanh(Cdt)

        if Hdtmn:  # skip connections
            Hdt = Hdt + Hdtmn

        return Hdt, Cdt


class ConvLSTM(nn.Module):
    def __init__(self,
                 input_filters: int,  # tell me the input at the top of the stack, I'll deal with the rest
                 hidden_filters: Union[int, List[int]],  # you can tell me all the sizes or just one
                 num_layers: int=1,
                 time_unroll: int=1,
                 skip_connect_n: int=None,
                 kernel: Union[int, tuple, List[int], List[tuple]]=3,  # I can see using this much more often tbh
                 stride: Union[int, tuple, List[int], List[tuple]]=1,
                 padding: Union[int, tuple, List[int], List[tuple]]=0
                 ):
        super(ConvLSTM, self).__init__()

        if isinstance(hidden_filters, list):
            assert isinstance(kernel, list)
            assert isinstance(stride, list)
            assert isinstance(padding, list)
            assert len(hidden_filters) == len(kernel) == len(stride) == len(padding) == num_layers

        # If kernel, stride, or padding is a list they must all be lists - otherwise just make it a number/tuple
        elif isinstance(kernel, list):
            assert isinstance(stride, list)
            assert isinstance(padding, list)
            assert len(kernel) == len(stride) == len(padding) == num_layers
            hidden_filters = [hidden_filters for i in range(num_layers)]

        elif isinstance(stride, list):
            assert isinstance(kernel, list)
            assert isinstance(padding, list)
            assert len(kernel) == len(stride) == len(padding) == num_layers
            hidden_filters = [hidden_filters for i in range(num_layers)]

        elif isinstance(padding, list):
            assert isinstance(stride, list)
            assert isinstance(kernel, list)
            assert len(kernel) == len(stride) == len(padding) == num_layers
            hidden_filters = [hidden_filters for i in range(num_layers)]

        elif num_layers != 1:
            hidden_filters = [hidden_filters for i in range(num_layers)]
            kernel = [kernel for i in range(num_layers)]
            stride = [stride for i in range(num_layers)]
            padding = [padding for i in range(num_layers)]

        else:
            assert not isinstance(kernel, list) and \
                   not isinstance(stride, list) and \
                   not isinstance(padding, list) and \
                   not isinstance(hidden_filters, list) and \
                   num_layers == 1

        self.stack = nn.ModuleList()
        self.num_layers = num_layers
        self.output_dims = ()
        self.time_unroll = time_unroll
        self.skip_connect_n = skip_connect_n
        if self.skip_connect_n:
            assert time_unroll >= skip_connect_n + 1

        for i in range(num_layers):
            self.stack.add_module("{}".format(i), ConvLSTMCell(input_filters,
                                                               hidden_filters[i],
                                                               kernel[i],
                                                               stride[i],
                                                               padding[i]))
            input_filters = hidden_filters[i]  # the next one should have input_filters == hidden_filters[i-1]

    def cuda(self, device_id=None):
        super(ConvLSTM, self).cuda(device_id)
        for i in range(self.num_layers):
            self.stack[i].cuda()

    def init_hidden(self,
                    batch_size: int,
                    input_dims: tuple  # you tell me the first input sizes, I'll take care of the rest.
                    ):

        H = []  # List along depth
        C = []  # List along depth
        h_size = input_dims
        for i in range(self.num_layers):  # add checks to make sure the sizes are still ok (by layer)
            # input size == hidden size of prev layer
            Hd, Cd, h_size = self.stack[i].init_hidden(batch_size, h_size)
            H.append(Hd)
            C.append(Cd)

        self.output_dims = h_size

        return H, C

    def forward(self, X0t, H, C):  # input is the same across all time steps

        # need a way to store previous H's (not just a big array)
        Htmn = []
        outputs = []  # this is of size [time_steps, X_output_size]
        for i in range(self.time_unroll):  # recurrent part
            Xdt = X0t  # input is the same across all time steps
            Htmn.append(H)  # hold onto previous H's, this will get a new entry for each time step
            for j in range(self.num_layers):  # depth of ConvLSTM
                if self.skip_connect_n and i > self.skip_connect_n:  # gotta be further along than skip_connect_n
                    H[j], C[j] = self.stack[j](Xdt, H[j], C[j],
                                               Htmn[i - self.skip_connect_n][j])  # Pass in the H from n time steps ago
                    Xdt = H[j]
                else:
                    H[j], C[j] = self.stack[j](Xdt, H[j], C[j])
                    Xdt = H[j]

            outputs.append(Xdt)  # I output the last depth at each time step, Xdt is a 4d tensor (list of 3d volumes)

        return outputs, H, C
