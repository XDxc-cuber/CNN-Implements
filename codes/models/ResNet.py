import torch
import torch.nn as nn


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False, activation=True):
        super(conv_bn_relu, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        if activation:
            self.seq.add_module("activation", nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.seq = nn.Sequential(
            conv_bn_relu(in_channels, out_channels, kernel_size, stride),
            conv_bn_relu(out_channels, out_channels, kernel_size, stride=1, activation=False),
        )
        if self.downsample:
            self.downsampleLayer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.seq(x)
        if self.downsample:
            out += self.downsampleLayer(x)
        else:
            out += x
        return self.activation(out)
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride, downsample=False, changeDim=False):
        super(BottleNeck, self).__init__()
        self.seq = nn.Sequential(
            conv_bn_relu(in_channels, mid_channels, 1, stride, 0),
            conv_bn_relu(mid_channels, mid_channels, kernel_size, 1),
            conv_bn_relu(mid_channels, out_channels, 1, 1, 0, activation=False),
        )
        xlayer_stride = 1
        if downsample:
            xlayer_stride = 2
        self.changeDim = changeDim
        if changeDim:
            self.xLayer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, xlayer_stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.seq(x)
        if self.changeDim:
            out += self.xLayer(x)
        else:
            out += x
        return self.activation(out)

class ResNet(nn.Module):
    def __init__(self, layers, num_classes=10, components="BasicBlock"):
        super(ResNet, self).__init__()
        if components != "BasicBlock" and components != "BottleNeck":
            raise ValueError("components must be BasicBlock or BottleNeck")
        
        # conv1
        self.seq = nn.Sequential(
            conv_bn_relu(1, 64, 7, 2, 3),
        )
        
        # conv2_x
        self.seq.append(nn.MaxPool2d(3, 2, 1))
        if components == "BasicBlock":
            self.seq.append(BasicBlock(64, 64, 3, 1, downsample=False))
        else:
            self.seq.append(BottleNeck(64, 256, 64, 3, 1, False, True))
        for i in range(1, layers[0]):
            if components == "BasicBlock":
                self.seq.append(BasicBlock(64, 64, 3, 1))
            else:
                self.seq.append(BottleNeck(256, 256, 64, 3, 1))
        
        # conv3_x
        self.add_block(components, 64, 256, layers[1])
        
        # conv4_x
        self.add_block(components, 128, 512, layers[2])
                
        # conv5_x
        self.add_block(components, 256, 1024, layers[3])
                
        # avg pool and fc
        self.seq.append(nn.AvgPool2d(7))
        self.seq.append(nn.Flatten())
        input_dims = 512 if components == "BasicBlock" else 2048
        self.seq.append(nn.Linear(input_dims, num_classes))
        
    def forward(self, x):
        return self.seq(x)
    
    def add_block(self, components, basicblockDim, bottleneckDim, num):
        if components == "BasicBlock":
            self.seq.append(BasicBlock(basicblockDim, basicblockDim*2, 3, 2, downsample=True))
        else:
            self.seq.append(BottleNeck(bottleneckDim, bottleneckDim*2, bottleneckDim//2, 3, 2, True, True))
        for i in range(1, num):
            if components == "BasicBlock":
                self.seq.append(BasicBlock(basicblockDim*2, basicblockDim*2, 3, 1))
            else:
                self.seq.append(BottleNeck(bottleneckDim*2, bottleneckDim*2, bottleneckDim//2, 3, 1))




def ResNet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes)
def ResNet34(num_classes=10):
    return ResNet([3, 4, 6, 3], num_classes)
def ResNet50(num_classes=10):
    return ResNet([3, 4, 6, 3], num_classes, "BottleNeck")
def ResNet101(num_classes=10):
    return ResNet([3, 4, 23, 3], num_classes, "BottleNeck")
def ResNet152(num_classes=10):
    return ResNet([3, 8, 36, 3], num_classes, "BottleNeck")

if __name__ == "__main__":
    inputs = torch.zeros((4, 1, 224, 224))
    model1 = ResNet18(10)
    model2 = ResNet34(10)
    model3 = ResNet50(10)
    model4 = ResNet101(10)
    model5 = ResNet152(10)
    models = [model1, model2, model3, model4, model5]
    
    for model in models:
        with torch.no_grad():
            output = model.forward(inputs)
            print(output.shape)
