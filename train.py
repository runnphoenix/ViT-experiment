from dataset import DoCaSet

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt

from timm import create_model 

train_data = DoCaSet(root_path='../cat_dog', category='train')
val_data = DoCaSet(root_path='../cat_dog', category='val')
test_data = DoCaSet(root_path = '../cat_dog', category='test')
print(len(train_data), len(val_data), len(test_data))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


model = create_model("vit_base_patch16_224", pretrained=False)
model.load_state_dict(torch.load("../jx_vit_base_p16_224-80ecf9dd.pth"))
model.head = torch.nn.Linear(model.head.in_features, 2)
torch.nn.init.xavier_uniform(model.head.weight)


for param in model.state_dict():
    model.state_dict()[param].requires_grad = False
    print(param, '\t', model.state_dict()[param].size())


model.state_dict()['head.weight'].requires_grad = True
model.state_dict()['head.bias'].requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
lr = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=20)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def train_epoch(model, optimizer):
    model.train()

    losses = []

    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        y = model(data)
        loss = criterion(y, label)

        losses.append(loss.cpu().item())
        loss.backward()

        optimizer.step()
        scheduler.step()

        if i > 0 and i % 500 == 0:
            print("Train: batch:{}, train_loss:{}".format(i, loss.cpu().item()))

    loss_avg = np.mean(losses)
    print("training loss for epoch: {}, lr:{}\n".format(loss_avg, scheduler.get_last_lr()[0]))
    return loss_avg


def val_epoch(model):
    model.eval()
    losses = []
    acc = 0.0

    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            y = model(data)

            loss = criterion(y, label)

            softmax = torch.nn.Softmax(dim=1)
            y = softmax(y.cpu())
            y = torch.argmax(y, dim=1)

            losses.append(loss.cpu().item())

            acc_batch = torch.mean((y==label.cpu()).float())
            acc += acc_batch

            if i > 0 and i % 100 == 0:
                print("validate: y:{}, label:{}".format(y, label))

    loss = np.mean(losses)
    acc = acc * 8.0 / len(val_data)
    print("Epoch val acc:{}%".format(acc))
    print("Epoch val loss:{}".format(loss))

    return acc

def write_result(results, file):
    with open(file, 'w') as f:
        for i in range(len(results)):
            f.write("{},{}\n".format(i, results[i]))

def test(model):
    model.eval()
    ys = []

    with torch.no_grad():
        for data in test_loader:
            # data
            data = data[0].to(device)
            # model 
            y = model(data).cpu()
            softmax = torch.nn.Softmax(dim=1)
            y = softmax(y)
            y = torch.argmax(y, dim=1)
            # write result to a file
            ys.extend(y)

    write_result(ys, './result.csv')


if __name__ == '__main__':

    losses = []
    val_accs = []

    for i in range(20):
        print("\n--------- Epoch {} ----------".format(i))
        # only for fine tune
        epoch_loss = train_epoch(model, optimizer)
        losses.append(epoch_loss)

        epoch_acc = val_epoch(model)
        val_accs.append(epoch_acc)
        
    plt.plot(range(20), losses)
    plt.plot(range(20), val_accs)
    plt.show()

    # test
    test(model)
