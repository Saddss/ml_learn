import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
y_data = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = Model()
loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

w_list = []
b_list = []
loss_list = []

for epoch in range(0, 1000):
    y_hat = model(x_data)
    l = loss(y_hat, y_data)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())
    loss_list.append(l.item())

# 创建画布和子图，包含3个子图
figure = plt.figure(figsize=(20, 6))

# 第一个子图：Loss vs Weight
ax1 = figure.add_subplot(1, 3, 1)
ax1.plot(w_list, loss_list, color='blue', linewidth=2, alpha=0.8)
ax1.set_title('Loss vs Weight (w)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Weight (w)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=10)

# 第二个子图：Loss vs Bias
ax2 = figure.add_subplot(1, 3, 2)
ax2.plot(b_list, loss_list, color='red', linewidth=2, alpha=0.8)
ax2.set_title('Loss vs Bias (b)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Bias (b)', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=10)

# 第三个子图：3D图 - Weight, Bias, Loss
ax3 = figure.add_subplot(1, 3, 3, projection='3d')

# 绘制训练轨迹的3D线条
ax3.plot(w_list, b_list, loss_list, color='green', linewidth=2, alpha=0.8, label='Training Path')

# 添加起始点和结束点标记
ax3.scatter(w_list[0], b_list[0], loss_list[0], color='red', s=100, label='Start', alpha=0.8)
ax3.scatter(w_list[-1], b_list[-1], loss_list[-1], color='blue', s=100, label='End', alpha=0.8)

# 设置标签和标题
ax3.set_xlabel('Weight (w)', fontsize=12)
ax3.set_ylabel('Bias (b)', fontsize=12)
ax3.set_zlabel('Loss', fontsize=12)
ax3.set_title('3D Training Path\n(Weight, Bias, Loss)', fontsize=14, fontweight='bold')

# 添加图例
ax3.legend()

# 设置视角（可以调整这些参数来改变3D图的视角）
ax3.view_init(elev=20, azim=45)

# 调整子图间距
plt.tight_layout()

# 添加整体标题
figure.suptitle('Linear Regression Training Process - 2D & 3D Views', fontsize=14, fontweight='bold', y=0.98)

# 显示图形
plt.show()

# 打印最终的参数值
print(f"Final weight: {w_list[-1]:.4f}")
print(f"Final bias: {b_list[-1]:.4f}")
print(f"Final loss: {loss_list[-1]:.4f}")