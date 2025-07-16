import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3])
y_data = np.array([2, 4, 6])

w_list = []
loss_list = []

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    temp = (y_pred - y) ** 2
    return temp.sum() / len(x)


if __name__ == '__main__':
    for w in np.arange(1, 3, 0.1):
        w_list.append(w)
        loss_val = loss(x_data, y_data)
        loss_list.append(loss_val)

    plt.plot(w_list, loss_list)
    plt.xlabel('w')
    plt.ylabel('loss')
    plt.show()