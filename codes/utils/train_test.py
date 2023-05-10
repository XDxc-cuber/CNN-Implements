import torch
import torch.nn.functional as F
import time

def loss_f(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)

def train(model, optimizer, train_data, valid_data, device, epochs, torch_resize):
    loss_history, val_acc_history = [], []
    model.to(device)
    
    for epoch in range(epochs):
        startTime = time.time()
        model.train()
        for x, y in train_data:
            x = torch_resize(x)
            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            del x
            loss = loss_f(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del y, y_pred
        
        if (epoch+1) % (epochs // 10) == 0:
            loss_sum, acc = get_acc(model, valid_data, device, torch_resize)
            loss_history.append(loss_sum)
            val_acc_history.append(acc)
            
            endTime = time.time()
            print("Epoch%d loss %f valid accuracy %f%% time cost %.4fmin" % (epoch+1, loss_sum, acc, (endTime - startTime) / 60))
            
    return loss_history, val_acc_history

def get_acc(model, data, device, torch_resize):
    model.eval()
    loss_sum = 0.
    num_true, num_total = 0, 0
    with torch.no_grad():
        for x, y in data:
            x = torch_resize(x)
            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            del x
            loss = loss_f(y_pred, y)
            loss_sum += loss.to('cpu')
            num_total += y.size(0)
            num_true += (torch.argmax(y_pred, dim=1) == y).sum().to('cpu')
            del y, y_pred
    
    return loss_sum, num_true / num_total * 100
