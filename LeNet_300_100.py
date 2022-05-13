
import torch.nn as nn


class LeNet_300_100(nn.Module):
    def __init__(self,cfg):
        if cfg == None:
            cfg=[300,100]
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, cfg[0])
        self.r_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(cfg[0], cfg[1])
        self.r_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(cfg[1], 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r_fc1(x)
        x = self.fc2(x)
        x = self.r_fc2(x)
        x = self.fc3(x)
        return x
