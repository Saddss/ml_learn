import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import re

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def clean_text(self, text):
        """清理文本"""
        # 处理NaN值和非字符串类型
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
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

class TestDataset(Dataset):
    def __init__(self, texts, phrase_ids, preprocessor):
        self.texts = texts
        self.phrase_ids = phrase_ids
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        phrase_id = self.phrase_ids[idx]
        
        sequence = self.preprocessor.text_to_sequence(text)
        
        return torch.tensor(sequence, dtype=torch.long), phrase_id

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

def predict():
    print("开始推理...")
    
    # 检查文件是否存在
    try:
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("预处理器加载完成")
    except FileNotFoundError:
        print("错误：未找到预处理器文件 'preprocessor.pkl'，请先运行训练脚本")
        return
    
    # 安全的设备检测 - 解决Mac M2 Pro + PyTorch 2.6兼容性问题
    def get_safe_device():
        """安全的设备检测，处理MPS兼容性问题"""
        try:
            # 检查MPS可用性和兼容性
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # PyTorch 2.6在某些Mac上可能有MPS bug，添加版本检查
                torch_version = torch.__version__
                print(f"🔍 检测到PyTorch版本: {torch_version}")

                # 测试MPS是否真正可用
                try:
                    test_tensor = torch.randn(10, 10, device='mps')
                    test_result = test_tensor + 1
                    del test_tensor, test_result
                    print("✅ MPS设备测试通过")
                    return torch.device('mps'), 0  # MPS时禁用多进程
                except Exception as e:
                    print(f"⚠️  MPS测试失败: {e}")
                    print("🔄 回退到CPU模式")
                    return torch.device('cpu'), 2
            elif torch.cuda.is_available():
                return torch.device('cuda'), 2
            else:
                return torch.device('cpu'), 2
        except Exception as e:
            print(f"⚠️  设备检测出错: {e}")
            print("🔄 使用CPU作为安全回退")
            return torch.device('cpu'), 2

    device, num_workers = get_safe_device()
    print(f"🚀 最终使用设备: {device}")
    
    # 创建模型
    model = SimpleRNN(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        output_dim=5,
        num_layers=1
    ).to(device)
    
    # 加载训练好的模型权重
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
        print("模型加载完成")
    except FileNotFoundError:
        print("错误：未找到模型文件 'best_model.pth'，请先运行训练脚本")
        return
    
    # 加载测试数据
    try:
        test_df = pd.read_csv('dataset/test.tsv', sep='\t')
        print(f"测试数据加载完成，共 {len(test_df)} 条数据")
    except FileNotFoundError:
        print("错误：未找到测试数据文件 'dataset/test.tsv'")
        return
    
    # 准备测试数据
    texts = test_df['Phrase'].tolist()
    phrase_ids = test_df['PhraseId'].tolist()
    
    # 创建测试数据集
    test_dataset = TestDataset(texts, phrase_ids, preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 进行预测
    predictions = []
    phrase_id_list = []
    
    print("开始预测...")
    with torch.no_grad():
        for batch_idx, (data, batch_phrase_ids) in enumerate(test_loader):
            data = data.to(device)
            
            # 模型预测
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # 收集预测结果
            predictions.extend(predicted.cpu().numpy())
            # 将tensor格式的phrase_ids转换为普通的Python整数
            if isinstance(batch_phrase_ids, torch.Tensor):
                phrase_id_list.extend(batch_phrase_ids.cpu().numpy().tolist())
            else:
                phrase_id_list.extend(batch_phrase_ids)
            
            if batch_idx % 100 == 0:
                print(f"已处理 {batch_idx * 64} / {len(test_dataset)} 条数据")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'PhraseId': phrase_id_list,
        'Sentiment': predictions
    })
    
    # 按PhraseId排序
    submission_df = submission_df.sort_values('PhraseId')
    
    # 保存预测结果
    output_file = 'submission.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f"预测完成！结果已保存到 {output_file}")
    print(f"预测结果统计:")
    print(submission_df['Sentiment'].value_counts().sort_index())
    
    # 显示前几行预测结果
    print("\n前10行预测结果:")
    print(submission_df.head(10))

if __name__ == "__main__":
    predict()