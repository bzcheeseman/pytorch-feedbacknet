#
# Created by Aman LaChapelle on 6/4/17.
#
# pytorch-feedbacknet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-feedbacknet/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ConvLSTM, Flatten
from tasks import train_data, test_data


class CIFARModel(nn.Module):
    def __init__(self, input_dims=(32, 32)):
        super(CIFARModel, self).__init__()

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        in_filters = 16
        hidden_filters = [32, 32, 64, 64]
        num_layers = 4
        time_unroll = 4
        skip_connect_n = 2
        kernels = [3, 3, 3, (3, 3)]
        strides = [2, 1, 2, (1, 1)]
        padding = [0, 0, 0, (0, 0)]

        self.convinput = (
            input_dims[0] - 3 + 1, input_dims[1] - 3 + 1
        )

        self.convlstm = ConvLSTM(in_filters, hidden_filters, num_layers, time_unroll,
                                 skip_connect_n, kernels, strides, padding)

        self.postprocess = nn.Sequential(
            nn.AvgPool2d(3, stride=1),
            Flatten(),
            nn.Linear(64, 100)
        )

    def cuda(self, device_id=None):
        super(CIFARModel, self).cuda(device_id)

        self.preprocess.cuda(device_id)
        self.convlstm.cuda(device_id)
        self.postprocess.cuda(device_id)

    def init_hidden(self, batch_size):
        return self.convlstm.init_hidden(batch_size, self.convinput)

    def forward(self, x, H, C):
        x = self.preprocess(x)
        IntRep, H, C = self.convlstm(x, H, C)
        outputs = []
        for x_t in IntRep:
            outputs.append(self.postprocess(x_t))

        return outputs, H, C  # List of predictions at each time step


def main(batch, max_epochs, print_steps):
    model = CIFARModel()
    model.cuda()

    try:
        model.load_state_dict(torch.load("checkpoints/checkpoint_cifar100.dat"))
    except FileNotFoundError:
        pass

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    # gammas = [0.3, 0.6, 0.9, 1.0]
    gammas = [1, 1, 1, 1]

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    running_loss = 0.0
    for epoch in range(max_epochs):
        for j, data in enumerate(train_loader, 0):
            input, target = data
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            H0, C0 = model.init_hidden(batch)

            opt.zero_grad()
            predicted, Hf, Cf = model(input, H0, C0)
            loss = 0.0
            for i, p in enumerate(predicted, 0):
                loss += gammas[i] * criterion(p, target)
            loss.backward()
            opt.step()

            running_loss += loss.data[0]

            if j % print_steps == print_steps-1:
                print("Epoch: {} - Step: {} - Avg Loss: {:.4f}".format(epoch+1, j+1, running_loss/print_steps))
                running_loss = 0.0
                torch.save(model.state_dict(), "checkpoints/checkpoint_cifar100.dat")

        running_loss = 0.0
        top1 = [0, 0, 0, 0]
        top5 = [0, 0, 0, 0]
        for j, data in enumerate(test_loader, 0):
            input, target = data
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            H0, C0 = model.init_hidden(1)

            predicted, Hf, Cf = model(input, H0, C0)
            loss = gammas[-1] * criterion(predicted[-1], target)
            running_loss += loss.data[0]

            for i, p in enumerate(predicted, 0):
                if torch.equal(Funct.softmax(p).data.topk(1)[1].squeeze(), target.data):
                    top1[i] += 1
                elif target.data[0] in Funct.softmax(p).data.topk(5)[1].squeeze().tolist():
                    top5[i] += 1

        print("Avg Loss Final Iteration: {:.4f}".format(running_loss/len(test_loader)))
        print("Top 1 Accuracy by Iteration: {} - Top 5 Accuracy by Iteration: {}".format(
            ["{:.4f}".format(t/len(test_loader)) for t in top1],
            ["{:.4f}".format(t/len(test_loader)) for t in top5])
        )
        running_loss = 0.0

if __name__ == '__main__':
    main(4, 10, 500)



