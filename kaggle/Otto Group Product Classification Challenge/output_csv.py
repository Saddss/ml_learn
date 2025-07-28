import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from train import Model
from train_improved import ImprovedModel

class TestDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.data = pd.read_csv(csv_file)
        self.ids = self.data['id'].values
        features = self.data.drop('id', axis=1).values.astype(np.float32)
        
        # 如果提供了scaler，使用它进行标准化
        if scaler is not None:
            features = scaler.transform(features)
            
        self.features = features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), self.ids[idx]

def main():
    # 加载模型和预处理器
    model = ImprovedModel()
    model.load_state_dict(torch.load('model_improved.pth', map_location='cpu'))
    # model = Model()
    # model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    
    # 加载测试数据（使用相同的标准化）
    test_dataset = TestDataset('./dataset/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 进行推理
    predictions = []
    test_ids = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, (features, ids) in enumerate(test_loader):
            # 前向传播
            outputs = model(features)

            # 应用softmax获得概率分布
            probabilities = torch.softmax(outputs, dim=1)

            # 收集结果
            predictions.extend(probabilities.cpu().numpy())
            test_ids.extend(ids.numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {(batch_idx + 1) * 64} samples...")
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    test_ids = np.array(test_ids)
    
    # 创建提交文件，确保格式与sampleSubmission.csv一致
    submission_df = pd.DataFrame()
    submission_df['id'] = test_ids.astype(int)
    
    # 添加类别概率列，保持与sampleSubmission.csv相同的列名格式
    for i in range(9):
        submission_df[f'Class_{i+1}'] = predictions[:, i]
    
    # 按id排序，确保顺序正确
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    # 保存结果，保留6位小数
    submission_df.to_csv('submission.csv', index=False, float_format='%.6f')
    print(f"Predictions saved to submission.csv")
    print(f"Shape: {submission_df.shape}")
    print("First few rows:")
    print(submission_df.head())

    # 验证概率和是否为1
    prob_sums = submission_df.iloc[:, 1:].sum(axis=1)
    print(f"Probability sums - Min: {prob_sums.min():.6f}, Max: {prob_sums.max():.6f}")
    print(f"Mean probability sum: {prob_sums.mean():.6f}")

if __name__ == "__main__":
    main()