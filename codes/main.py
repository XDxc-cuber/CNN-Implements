import torch
from torchvision import transforms
import numpy as np
import argparse

from models import AlexNet, LeNet, ResNet, VGG16, MobileNetV2
from utils import loadData, train_test


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    """
        recommended arguments(recommended only, according to GPU mem):
            LeNet:                 0.02 min/epoch
                batch_size    512
            AlexNet:               3 min/epoch
                batch_size    128
            ResNet:                12.8 min/epoch
                batch_size    64
            VGG-16:                VERYLARGE min/epoch
                batch_size    8
            MobileNetV2:           6.6 min/epoch
                batch_size    32
    """
    parser = argparse.ArgumentParser(description="Command")
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--reg', default=0, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default="LeNet", type=str)
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    
    args = parser.parse_args()
    
    # set random seed
    setup_seed(666)
    
    
    model = None
    optimizer = None
    data_path = "codes/dataset"
    pic_shape = (0, 0)
    
    
    # get model
    if args.model == 'LeNet':
        model = LeNet.LeNet()
        pic_shape = (32, 32)
    elif args.model == 'AlexNet':
        model = AlexNet.AlexNet(dropout_p=args.dropout)
        pic_shape = (224, 224)
    elif args.model == 'ResNet':
        model = ResNet.ResNet34()
        pic_shape = (224, 224)
    elif args.model == 'VGG':
        model = VGG16.VGG16(dropout_p=args.dropout)
        pic_shape = (224, 224)
    elif args.model == 'MobileNet':
        model = MobileNetV2.MobileNetV2()
        pic_shape = (224, 224)
    else:
        raise ValueError('Unknown model')
    
    
    # get optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise ValueError("Unknown optimizer")
    
    
    # load data
    print("Loading data...")
    train_data, valid_data, test_data = loadData.loadData(data_path, batch_size=args.batch_size)
    print()
    
    # train
    print("Training...")
    print("Model: %s"%(args.model))
    torch_resize = transforms.Resize(pic_shape)
    loss_his, val_his = train_test.train(model, optimizer, train_data, valid_data, args.device, args.epoch, torch_resize)
    
    # test
    print("Testing...")
    loss_sum, test_acc = train_test.get_acc(model, test_data, args.device, torch_resize)
    print("Test loss %f acc %.4f%%" % (loss_sum, test_acc))
    
    # print history
    # print(loss_his, val_his)
    
    
