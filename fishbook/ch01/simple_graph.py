# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


a = np.array([[1, 2], [3, 4]])
for element in a.flatten():
    print(element)

print(a > 2)
print(a[a > 2])

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()