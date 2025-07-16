import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.FloatTensor([[1], [2], [3]])
y_data = torch.FloatTensor([[0], [0], [1]])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

w_list = []
b_list = []
loss_list = []

for epoch in range(0, 1000):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())
    loss_list.append(loss.item())

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()