import torch
from torch import nn
from torch.nn import init


class VGGNet2016(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, (3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, (3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),
        )

        self.n_flat = 64 * 1 * 55
        self.linear = nn.Sequential(
            nn.Linear(self.n_flat, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 88)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        init.xavier_uniform_(self.linear[-1].weight, 1)

    # returns logits!
    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, self.n_flat)
        h = self.linear(h)
        return h

    # returns pseudo probabilities
    def predict(self, x):
        return torch.sigmoid(self.forward(x))
