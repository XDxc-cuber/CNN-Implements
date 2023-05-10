import torch
import torch.nn as nn


class MobileNetBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride):
        super(MobileNetBottleNeck, self).__init__()
        expansion = in_channels * t
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, expansion, 1),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expansion, expansion, 3, groups=expansion, stride=stride, padding=1),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expansion, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.do_shortcut = False
        if stride == 1:
            self.do_shortcut = True
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Sequential()
    def forward(self, x):
        out = self.seq(x)
        if self.do_shortcut:
            out = out + self.shortcut(x)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
        )
        self.add_bottleneck(1, 32, 16, 1, 1)
        self.add_bottleneck(6, 16, 24, 2, 2)
        self.add_bottleneck(6, 24, 32, 3, 2)
        self.add_bottleneck(6, 32, 64, 4, 2)
        self.add_bottleneck(6, 64, 96, 3, 1)
        self.add_bottleneck(6, 96, 160, 3, 2)
        self.add_bottleneck(6, 160, 320, 1, 1)
        self.seq.append(nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1),
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(1280, num_classes),
        ))
    def add_bottleneck(self, t, in_channels, out_channels, n, stride):
        self.seq.append(MobileNetBottleNeck(in_channels, out_channels, t, stride))
        for i in range(n-1):
            self.seq.append(MobileNetBottleNeck(out_channels, out_channels, t, 1))
    def forward(self, x):
        return self.seq(x)
        
if __name__ == '__main__':
    x = torch.zeros((3, 1, 224, 224))
    labels = torch.zeros((3)).long()
    model = MobileNetV2(num_classes=10)
    y = model.forward(x)
    print(y.shape)
