import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, dropout_p=0.5, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    x = torch.zeros((3, 1, 224, 224))
    labels = torch.zeros((3)).long()
    model = AlexNet(num_classes=10)
    y = model.forward(x)
    print(y.shape)


