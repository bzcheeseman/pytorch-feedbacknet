#
# Created by Aman LaChapelle on 6/4/17.
#
# pytorch-feedbacknet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-feedbacknet/LICENSE.txt
#

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR100("data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR100("data", train=False, download=True, transform=transform)
