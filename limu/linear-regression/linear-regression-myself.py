import torch
from torch import nn
from torch.utils import data

def gen_data(true_w, true_b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(true_w)))
    Y = torch.matmul(X, true_w) + true_b
    Y += torch.normal(0, 0.01, Y.shape)
    return X, Y.reshape(-1, 1)

class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

features, labels = gen_data(torch.tensor([2, 3], dtype=torch.float32), torch.tensor(1), 1000)
batch_size = 10
train_iter = data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=True)

linear_model = LinearRegression(2, 1)
linear_model.linear.weight.data.normal_(0, 0.01)
linear_model.linear.bias.data.fill_(0)

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
