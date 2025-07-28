import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from train import Model
from train_improved import ImprovedModel

class DataSet(Dataset):
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)
        self.x_data = np.loadtxt('./dataset/train.csv', delimiter=',', dtype=object, skiprows=1)
        self.x_test_data = torch.from_numpy(self.x_data[50000:, 1:-1].astype(np.float32))
        y_test_labels = self.x_data[50000:, -1]
        y_test_numbers = np.array([int(str(label).split('_')[-1]) - 1 for label in y_test_labels])
        self.y_test_data = torch.from_numpy(y_test_numbers.astype(np.int64))
        self.len = self.x_test_data.shape[0]

    def __getitem__(self, index):
        return self.x_test_data[index], self.y_test_data[index]

    def __len__(self):
        return self.len

dataset = DataSet()
test_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# 正确的模型加载方式
# model = Model()
model = ImprovedModel()
model.load_state_dict(torch.load('model_improved.pth', map_location='cpu'))
# model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

with torch.no_grad():
    acc_sum = 0
    for (inputs, outputs) in test_loader:
        y_hat = model(inputs)
        y_target = y_hat.argmax(dim=1)
        acc_num = (outputs == y_target).sum().item()
        acc_sum += acc_num
    print('acc: {}'.format(acc_sum / len(dataset)))