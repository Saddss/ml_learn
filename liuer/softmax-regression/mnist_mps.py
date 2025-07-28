import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.seq(x)
        return x

model = Model().to(device)
# model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("../dataset", train=True, download=True, transform=transform)
test_data = datasets.MNIST("../dataset", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, outputs) in enumerate(train_loader):
        # 放到mps上使用gpu加速
        inputs, outputs = inputs.to(device), outputs.to(device)
        y_hat = model(inputs)
        loss = criterion(y_hat, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print("Epoch: {}, Loss: {}".format(epoch, running_loss))

def test():
    acc_sum = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for (inputs, outputs) in test_loader:
            # 放到mps上使用gpu加速, 加non_blocking=True会导致数据错乱
            inputs, outputs = inputs.to(device), outputs.to(device)
            y_hat = model(inputs)
            # y_target = y_hat.argmax(axis=1)
            _, y_target = torch.max(y_hat.data, dim = 1)
            acc_sum += (outputs == y_target).sum().item()
            print(outputs)
            total += len(outputs)
        print("Accuracy: {}".format(acc_sum / total))

if __name__ == '__main__':
    for i in range(100):
        train(i)
        if i != 0 and i % 10 == 0 or i == 99:
            test()
