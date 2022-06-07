from dataset import DoCaSet

import torch
from torch.utils.data import DataLoader


from timm import create_model as creat

train_data = DoCaSet(root_path='../cat_dog', category='train')
val_data = DoCaSet(root_path='../cat_dog', category='val')
test_data = DoCaSet(root_path = '../cat_dog', category='test')

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


model = creat("vit_base_patch16_224", pretrained=False)
    # fix weights
model.load_state_dict(torch.load("../jx_vit_base_p16_224-80ecf9dd.pth"))
model.head = torch.nn.Linear(model.head.in_features, 2)

for param in model.state_dict():
    print(param, '\t', model.state_dict()[param].size())


criterion = torch.nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def train_epoch(model, optimizer, dataset):
    model.train()

    losses = []

    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        y = model(data)
        loss = criterion(y, label)

        losses.append(loss)
        loss.backward()

        optimizer.step()

        loss_avg = np.mean(losses)


def val_epoch(model, dataset):
    model.eval()
    losses = []
    ys = []

    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            y = model(data)
            loss = criterion(y, label)
            losses.append(loss)
            ys.append(y)

    loss = np.mean(losses)
    acc = (val_data[1] == ys).mean()

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
            print(y)
            y = torch.argmax(y, dim=1)
            print(y)
            # write result to a file
            ys.extend(y)

    write_result(ys, './result.csv')


if __name__ == '__main__':
    # only for fine tune
    #train()
    #val()

    # test
    test(model)
