from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x = self.sigmoid(self.linear1(x))
        # x = self.sigmoid(self.linear2(x))
        # x = self.sigmoid(self.linear3(x))

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

class MyDataset(Dataset):
    def __init__(self, filepath):
        self.data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        
        # 数据归一化 - 这很重要！
        self.data[:, :-1] = (self.data[:, :-1] - self.data[:, :-1].mean(axis=0)) / self.data[:, :-1].std(axis=0)
        
        self.x_train_data = torch.from_numpy(self.data[:600, :-1])
        self.y_train_data = torch.from_numpy(self.data[:600, [-1]])
        self.x_test_data = torch.from_numpy(self.data[600:, :-1])
        self.y_test_data = torch.from_numpy(self.data[600:, [-1]])
        self.len = self.x_train_data.shape[0]

    def __getitem__(self, index):
        return self.x_train_data[index], self.y_train_data[index]

    def __len__(self):
        return self.len

def calculate_accuracy(model, x_data, y_data, threshold=0.5):
    """
    计算准确率
    Args:
        model: 训练好的模型
        x_data: 输入数据
        y_data: 真实标签
        threshold: 分类阈值，默认0.5
    Returns:
        accuracy: 准确率
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        y_pred = model(x_data)
        # 根据阈值进行二分类判断
        predictions = (y_pred >= threshold).float()
        correct = (predictions == y_data).float()
        # correct = correct[predictions == y_data].float()
        # print(correct)
        accuracy = correct.sum() / len(y_data)
    model.train()  # 设置回训练模式
    return accuracy.item()

data = MyDataset("../study/diabetes.csv")
dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)
model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 提高学习率
epoch = 500

# 可调整的阈值
threshold = 0.5

if __name__ == '__main__':
    print(f"使用阈值: {threshold}")
    print("数据集大小: 训练集={}, 测试集={}".format(len(data.x_train_data), len(data.x_test_data)))
    print("Epoch | Train Loss | Train Acc | Test Acc")
    print("-" * 45)
    
    for i in range(epoch):
        # 训练阶段
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader, 0):
            y_hat = model(inputs)
            loss = criterion(y_hat, labels)
            total_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 计算平均训练损失 - 修复了这个bug!
        avg_train_loss = total_loss / num_batches
        
        # 计算训练集准确率
        train_accuracy = calculate_accuracy(model, data.x_train_data, data.y_train_data, threshold)
        
        # 计算测试集准确率
        test_accuracy = calculate_accuracy(model, data.x_test_data, data.y_test_data, threshold)
        
        # 每10轮打印一次结果
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i+1:5d} | {avg_train_loss:10.4f} | {train_accuracy:9.4f} | {test_accuracy:8.4f}")
    
    print("-" * 45)
    print(f"最终训练集准确率: {train_accuracy:.4f}")
    print(f"最终测试集准确率: {test_accuracy:.4f}")