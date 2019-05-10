import torch.nn as nn

def CBR(ic, rc, kernel_size, stride):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(ic, rc, kernel_size=kernel_size, padding=pad, stride=stride),
        nn.BatchNorm2d(rc),
        nn.ReLU())

class MiniModel(nn.Module):
    def __init__(self):
        super(MiniModel, self).__init__()

        self.feature_extraction = nn.Sequential(
            CBR(3, 32, 3, 1),
            CBR(32, 64, 3, 2),
            CBR(64, 256, 3, 1),
            CBR(256, 256, 3, 2))
        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 10))

    def forward(self, x):
        B, C, H, W = x.shape
        # f = Bx256x7x7
        f = self.feature_extraction(x)
        return self.fully_connected(f.view(B, -1))
