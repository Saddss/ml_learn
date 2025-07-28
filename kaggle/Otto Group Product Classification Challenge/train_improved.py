import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class DataSet(Dataset):
    def __init__(self, features, labels, train=True, scaler=None):
        super(DataSet, self).__init__()
        self.train = train
        
        # if scaler is None:
        #     self.scaler = StandardScaler()
        #     self.features = torch.from_numpy(self.scaler.fit_transform(features).astype(np.float32))
        # else:
        #     self.scaler = scaler
        #     self.features = torch.from_numpy(self.scaler.transform(features).astype(np.float32))
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        
    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        
        # 数据增强：仅在训练时添加噪声
        if self.train and torch.rand(1) < 0.5:
            noise = torch.randn_like(features) * 0.02
            features = features + noise
            
        return features, label

    def __len__(self):
        return len(self.features)

class ImprovedModel(nn.Module):
    def __init__(self, input_dim=93, num_classes=9):
        super().__init__()
        
        # 大幅简化模型结构
        self.layers = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),

            # 第二层  
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),

            # 第三层
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # 输出层
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total, total_loss / len(data_loader)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    data = np.loadtxt('./dataset/train.csv', delimiter=',', dtype=object, skiprows=1)
    features = data[:, 1:-1].astype(np.float32)
    labels_str = data[:, -1]
    labels = np.array([int(str(label).split('_')[-1]) - 1 for label in labels_str])
    
    print(f"Data shape: {features.shape}, Labels shape: {labels.shape}")
    
    # 划分训练集和验证集 (80:20)
    total_size = len(features)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 随机划分
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    # 创建数据集
    train_dataset = DataSet(train_features, train_labels, train=True)
    # val_dataset = DataSet(val_features, val_labels, train=False, scaler=train_dataset.scaler)
    val_dataset = DataSet(val_features, val_labels, train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 创建模型
    model = ImprovedModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True, min_lr=1e-6
    )

    # 训练参数
    num_epochs = 200
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        
        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  LR: {current_lr:.6f}, Gap: {train_acc - val_acc:.4f}')
        
        # 早停和模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'model_improved.pth')
            print(f'  ✓ New best validation accuracy: {best_val_acc:.4f}')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
        
        print('-' * 50)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    # 加载最佳模型并最终评估
    model.load_state_dict(torch.load('model_improved.pth', map_location=device))

    final_train_acc, _ = evaluate_model(model, train_loader, device)
    final_val_acc, _ = evaluate_model(model, val_loader, device)
    
    print(f'\nFinal Results:')
    print(f'Training Accuracy: {final_train_acc:.4f}')
    print(f'Validation Accuracy: {final_val_acc:.4f}')
    print(f'Overfitting Gap: {final_train_acc - final_val_acc:.4f}')

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()