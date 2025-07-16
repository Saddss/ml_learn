import torch
from torch import nn
from torch.utils import data

def gen_data(true_w, true_b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(true_w)))
    Y = torch.matmul(X, true_w) + true_b
    Y += torch.normal(0, 0.01, Y.shape)
    return X, Y.reshape(-1, 1)

features, labels = gen_data(torch.tensor([2, 3, 4], dtype=torch.float32), torch.tensor(1), 1000)
batch_size = 10
train_iter = data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=True)

linear_model = nn.Sequential(nn.Linear(3, 1))
linear_model[0].weight.data.normal_(0, 0.01)
linear_model[0].bias.data.fill_(0)

loss = nn.MSELoss()

optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(linear_model(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(linear_model(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
