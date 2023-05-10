import torch
import torch.nn as nn


class VGG_block(nn.Module):
    def __init__(self, conv_num, in_channels, out_channels):
        super(VGG_block, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        for i in range(1, conv_num):
            self.seq.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            self.seq.append(nn.ReLU(inplace=True))
        self.seq.append(nn.MaxPool2d(2, 2))
    def forward(self, x):
        return self.seq(x)

class VGG16(nn.Module):
    def __init__(self, dropout_p=0.5, num_classes=10):
        super(VGG16, self).__init__()
        self.seq = nn.Sequential(
            VGG_block(2, 1, 64),
            VGG_block(2, 64, 128),
            VGG_block(3, 128, 256),
            VGG_block(3, 256, 512),
            VGG_block(3, 512, 512),
            
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    x = torch.zeros((3, 1, 224, 224))
    labels = torch.zeros((3)).long()
    model = VGG16(num_classes=10)
    y = model.forward(x)
    print(y.shape)
