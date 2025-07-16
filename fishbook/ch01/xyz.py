import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建网格
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

Z = 1 - X - Y  # 根据方程 x + y + z = 1
print(Z)

# 绘制平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')

# 绘制截距点
ax.scatter(1, 0, 0, color='red', label='(1,0,0)')
ax.scatter(0, 1, 0, color='green', label='(0,1,0)')
ax.scatter(0, 0, 1, color='purple', label='(0,0,1)')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
# plt.show()
