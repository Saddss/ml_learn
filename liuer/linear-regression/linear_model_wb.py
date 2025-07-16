import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 3, 5, 7])

def forward(x):
    return x * w + b

def loss(x, y):
    y_hat = forward(x)
    temp = (y_hat - y) ** 2
    return temp.sum()

w_list = []
b_list = []
avg_loss_list = []

if __name__ == '__main__':
    for w, b in zip(np.arange(1, 3, 0.1), np.arange(0, 2, 0.1)):
        w_list.append(w)
        b_list.append(b)
        print('w:', w, 'b:', b)
        y_data_hat = forward(x_data)
        current_loss = loss(x_data, y_data) / len(x_data)
        avg_loss_list.append(current_loss)

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制 loss 随 w 变化的图像
    ax1.plot(w_list, avg_loss_list)
    ax1.set_title('Loss vs w')
    ax1.set_xlabel('w')
    ax1.set_ylabel('Loss')
    
    # 绘制 loss 随 b 变化的图像
    ax2.plot(b_list, avg_loss_list)
    ax2.set_title('Loss vs b')
    ax2.set_xlabel('b')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()