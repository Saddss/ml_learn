import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

class DataSet(Dataset):
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)
        self.x_data = np.loadtxt('./dataset/train.csv', delimiter=',', dtype=object, skiprows=1)
        self.x_train_data = torch.from_numpy(self.x_data[:50000, 1:-1].astype(np.float32))
        y_train_labels = self.x_data[:50000, -1]
        y_train_numbers = np.array([int(str(label).split('_')[-1]) - 1 for label in y_train_labels])
        self.y_train_data = torch.from_numpy(y_train_numbers.astype(np.int64))
        self.len = self.x_train_data.shape[0]

    def __getitem__(self, index):
        return self.x_train_data[index], self.y_train_data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(93, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.seq(x)

if __name__ == "__main__":
    dataset = DataSet()
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = 100

    for i in range(epoch):
        loss_sum = 0.0
        acc_sum = 0.0
        for (inputs, outputs) in train_loader:
            y_hat = model(inputs)
            loss = criterion(y_hat, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            y_target = y_hat.argmax(dim=1)
            acc_num = (outputs == y_target).sum().item()
            acc_sum += acc_num
        print('epoch : {}, loss: {}, acc: {}'.format(i, loss_sum, acc_sum/len(dataset)))
        if i == 99:
            torch.save(model.state_dict(), 'model.pth')