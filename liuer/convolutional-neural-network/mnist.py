import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        print(x.size(0))
        x = self.seq(x)
        return x
    
train_data = torchvision.datasets.MNIST(root='../dataset', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='../dataset', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = Model()
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch = 100

for i in range(epoch):
    train_loss_sum = 0
    train_acc_sum = 0
    for (inputs, outputs) in train_loader:
        y_hat = model(inputs)
        loss = criterion(y_hat, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc_sum += loss.item()
        train_acc_sum += (outputs == y_hat.argmax(dim=1)).sum().item()
    print(f'epoch : {i}, loss: {train_loss_sum/len(train_loader)}, acc: {train_acc_sum/len(train_data)}')
    test_loss_sum = 0
    test_acc_sum = 0
    with torch.no_grad():
        for (inputs, outputs) in test_loader:
            y_hat = model(inputs)
            loss = criterion(y_hat, outputs)
            test_acc_sum += loss.item()
            test_acc_sum += (outputs == y_hat.argmax(dim=1)).sum().item()
        print(f'epoch : {i}, loss: {test_loss_sum/len(test_loader)}, acc: {test_acc_sum/len(test_data)}')
