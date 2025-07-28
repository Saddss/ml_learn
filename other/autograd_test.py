import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
label = torch.tensor([1, 4, 9], dtype=torch.float32)
w = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
y = w * x
print(y.grad_fn._saved_self)
y.backward()
print(x.grad)

