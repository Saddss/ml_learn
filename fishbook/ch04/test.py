import numpy as np

a = np.array([[0.6,0.4], [0.3,0.7], [0.5,0.5]])
b = np.array([[1, 0], [0, 1], [0, 1]])
c = np.array([0, 1, 1])

# print(-b * np.log(a))
# print(np.sum(-b * np.log(a), axis=1, keepdims=True))
#
# print(a[np.array(range(3)), c])

print(a[0][:])