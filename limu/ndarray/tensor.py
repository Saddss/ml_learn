import torch, pandas as pd, numpy as np

# a = torch.randint(0, 10, (1, 3))
# b = torch.randint(0, 10, (3, 1))
# print(a + b)

A = torch.arange(20).reshape(4, 5)
B = torch.arange(20).reshape(5, 4)
# print(A==B)

print(torch.mm(A, B))
print(np.array(A).cumsum())
