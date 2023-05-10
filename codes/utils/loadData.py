import numpy as np
import struct
import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def decode_idx3_ubyte(file):
    with open(file, 'rb') as f:
        bin_data = f.read()
    offset = 0
    fmt_header = '>iiii'
    magic, numImgs, numRows, numCols = struct.unpack_from(fmt_header, bin_data, offset)
    
    offset = struct.calcsize(fmt_header)
    fmt_image = '>' + str(numImgs * numRows * numCols) + 'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).reshape(numImgs, 1, numRows, numCols).float()
    return data

def decode_idx1_ubyte(file):
    with open(file, 'rb') as f:
        bin_data = f.read()
    offset = 0
    fmt_header = '>ii'
    magic, numImgs = struct.unpack_from(fmt_header, bin_data, offset)
    
    offset = struct.calcsize(fmt_header)
    fmt_image = '>' + str(numImgs) + 'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).long()
    return data

def loadData(path, valid_rate=0.1, batch_size=256):
    x_train = decode_idx3_ubyte(path + "/train_data")
    y_train = decode_idx1_ubyte(path + "/train_labels")
    x_test = decode_idx3_ubyte(path + "/test_data")
    y_test = decode_idx1_ubyte(path + "/test_labels")
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_rate, random_state=666)
    print("train size: ", x_train.shape)
    print("valid size: ", x_valid.shape)
    print("test size: ", x_test.shape)
    
    train_data = Data.TensorDataset(x_train, y_train)
    train_data = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = Data.TensorDataset(x_valid, y_valid)
    valid_data = Data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data = Data.TensorDataset(x_test, y_test)
    test_data = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_data, valid_data, test_data
    
        
        
if __name__ == "__main__":
    train_data, valid_data, test_data = loadData("dataset")
    print(train_data.batch_sampler)

