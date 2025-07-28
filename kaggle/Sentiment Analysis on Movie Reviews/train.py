import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import pickle
from collections import Counter

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class TextPreprocessor:
    def __init__(self, max_vocab_size=100000, max_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def clean_text(self, text):
        """清理文本"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def build_vocab(self, texts):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            words = self.clean_text(text).split()
            word_counts.update(words)
        
        # 保留最常见的词汇
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # 构建词汇映射
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(most_common, 2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        self.vocab_size = len(self.word_to_idx)
        print(f"构建词汇表完成，词汇量: {self.vocab_size}")
    
    def text_to_sequence(self, text):
        """将文本转换为序列"""
        words = self.clean_text(text).split()
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1是<UNK>的索引
        
        # 截断或填充
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([0] * (self.max_length - len(sequence)))  # 0是<PAD>的索引
            
        return sequence

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        sequence = self.preprocessor.text_to_sequence(text)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=5, num_layers=1):
        super(SimpleRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM输出
        output, (hidden, cell) = self.rnn(embedded)
        
        # 使用最后一个时间步的隐藏状态
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # 应用dropout和全连接层
        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch_size, output_dim)
        
        return out

def train_model():
    print("开始加载数据...")
    
    # 加载训练数据
    train_df = pd.read_csv('dataset/train.tsv', sep='\t')
    print(f"训练数据加载完成，共 {len(train_df)} 条数据")
    
    # 数据预处理
    texts = train_df['Phrase'].tolist()
    labels = train_df['Sentiment'].tolist()
    
    print("情感标签分布:")
    print(train_df['Sentiment'].value_counts().sort_index())
    
    # 构建预处理器
    preprocessor = TextPreprocessor(max_vocab_size=100000, max_length=50)
    preprocessor.build_vocab(texts)
    
    # 保存预处理器
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("预处理器已保存")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据集
    train_dataset = SentimentDataset(X_train, y_train, preprocessor)
    val_dataset = SentimentDataset(X_val, y_val, preprocessor)

    # 创建数据加载器
    batch_size = 64
    # Mac优化：减少或禁用多进程数据加载
    num_workers = 0 if torch.backends.mps.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 创建模型 - Mac MPS优化设备检测
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🚀 使用 Apple Silicon MPS 加速: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用 CUDA GPU 加速: {device}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  使用 CPU (较慢): {device}")
    
    model = SimpleRNN(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        output_dim=5,
        num_layers=1
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 100
    best_val_acc = 0
    
    print("开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 500 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print('-' * 50)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()