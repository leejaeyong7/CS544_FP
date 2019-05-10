import torch.nn as nn

def CBR(ic, rc, kernel_size, stride):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(ic, rc, kernel_size=kernel_size, padding=pad, stride=stride),
        nn.BatchNorm2d(rc),
        nn.ReLU())

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 30),
            nn.Linear(30, 10))

    def forward(self, x):
        B, C, H, W = x.shape
        # f = Bx256x7x7
        return self.fc(x.view(B, -1))
