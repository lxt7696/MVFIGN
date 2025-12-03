"""
超简单推理速度测试脚本
"""

import torch
import time
import numpy as np
from mul_model import GMLP
from mul_deal_trans import deal_dataset

# ==================== 配置区域 ====================
# 在这里修改所有参数
MODEL_PATH = './saved_models/MIGN_PubMed_best_r1234.pth'  # 你的模型路径
DATASET = 'PubMed'
R = 4  # 你训练时用的 r 值（1, 2, 3, 4, 等）
HIDDEN = 512
DROPOUT = 0.6
NUM_RUNS = 100  # 测试次数
WARMUP = 10     # 预热次数
# ==================================================


def test_inference_speed():
    """测试推理速度"""
    print("="*60)
    print("推理速度测试")
    print("="*60)
    
    # 1. 加载数据
    print("\n正在加载数据...")
    user_features, user_labels, client_adj, label_num, _, _, \
    test_features, test_labels, val_features, val_labels, at_f, at_l = \
        deal_dataset(client_num=1, rate=1.0, r=R)
    
    feature_dim = len(user_features[0][0])
    print(f" 测试样本数量: {len(test_labels)}")
    print(f" 特征维度: {feature_dim}")
    
    # 2. 加载模型
    print("\n正在加载模型...")
    model = GMLP(r=R, nfeat=feature_dim, nhid=HIDDEN, 
                 nclass=label_num, dropout=DROPOUT, drate=1.0)
    
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_features = test_features.to(device)
    model.eval()
    
    print(f" 模型已加载")
    print(f" 使用设备: {device}")
    print(f" 模型使用 {R} 个 hop")
    
    # 3. 测试准确率
    print("\n正在测试准确率...")
    with torch.no_grad():
        output = model(test_features)
        preds = output.max(1)[1]
        correct = preds.eq(torch.tensor(test_labels).to(device)).sum().item()
        accuracy = (correct / len(test_labels)) * 100
    print(f" 测试准确率: {accuracy:.2f}%")
    
    # 4. 预热
    print(f"\n正在预热 ({WARMUP} 次)...")
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(test_features)
            
    # 预热后等待一下
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        time.sleep(0.5)  # 等待0.5秒
    
    # 5. 测试推理速度
    print(f"\n正在测试推理速度 ({NUM_RUNS} 次)...")
    times = []
    
    with torch.no_grad():
        for i in range(NUM_RUNS):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(test_features)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # 转为毫秒
    
    # 6. 统计结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 7. 输出结果
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"准确率:        {accuracy:.2f}%")
    print(f"平均时间:      {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"最小时间:      {min_time:.2f} ms")
    print(f"最大时间:      {max_time:.2f} ms")
    print(f"吞吐量:        {len(test_labels)/avg_time*1000:.0f} samples/sec")
    print("="*60)

    
    return accuracy, avg_time


if __name__ == '__main__':
    test_inference_speed()