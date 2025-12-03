import torch
import time
import numpy as np
from mul_model import GMLP
from mul_deal_trans_r2 import deal_dataset

def test_inference_speed(model, test_features, test_labels, num_runs=100, warmup=10):
    """
    测试模型推理速度
    
    Args:
        model: 训练好的模型
        test_features: 测试集特征
        test_labels: 测试集标签
        num_runs: 测试运行次数
        warmup: 预热次数（排除初始化开销）
    
    Returns:
        avg_time_ms: 平均推理时间（毫秒）
        accuracy: 准确率
    """
    model.eval()
    
    # 预热阶段
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(test_features)
    
    # 正式测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            # 同步GPU（如果使用GPU）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(test_features)
            
            # 同步GPU确保计算完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算准确率
    with torch.no_grad():
        output = model(test_features)
        preds = output.max(1)[1].type_as(torch.tensor(test_labels))
        correct = preds.eq(torch.tensor(test_labels)).sum().item()
        accuracy = correct / len(test_labels)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, accuracy


def load_and_test_model(model_path, dataset='PubMed', r=2):
    """
    加载模型并测试推理速度
    """
    # 加载数据
    user_features, user_labels, client_adj, label_num, train_features, train_labels, \
    test_features, test_labels, val_features, val_labels, at_f, at_l = \
        deal_dataset(client_num=1, rate=1.0, r=r)
    
    feature_dim = len(user_features[0][0])
    
    # 创建模型
    model = GMLP(
        r=r,
        nfeat=feature_dim,
        nhid=512,
        nclass=label_num,
        dropout=0.6,
        drate=1.0
    )
    
    # 加载训练好的模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if torch.cuda.is_available():
        model = model.cuda()
        test_features = test_features.cuda()
    
    # 测试推理速度
    avg_time, std_time, accuracy = test_inference_speed(
        model, test_features, test_labels, num_runs=100, warmup=10
    )
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}")
    print(f"Model: MIGN (r={r})")
    print(f"{'='*50}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Average Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {len(test_labels)/avg_time*1000:.2f} samples/second")
    print(f"{'='*50}\n")
    
    return avg_time, accuracy


if __name__ == '__main__':
    # 方法1：直接测试保存的模型
    load_and_test_model(
        model_path='./saved_models/MIGN_PubMed_best_r12.pth',
        dataset='PubMed',
        r=2
    )
    
    # 方法2：完整对比实验
    # compare_with_baselines()