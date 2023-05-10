import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__ (self, num_classes=10):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )
        
    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x
    
    
if __name__ == '__main__':
    x = torch.zeros((3, 1, 32, 32))
    labels = torch.zeros((3)).long()
    model = LeNet(num_classes=10)
    y = model.forward(x)
    print(y.shape)
    
    
    
    
    
    