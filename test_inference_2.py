"""
增强版推理速度和模型性能测试脚本
"""

import torch
import time
import numpy as np
import os
#import psutil
from mul_model import GMLP
from mul_deal_trans_r23456 import deal_dataset

# ==================== 配置区域 ====================
MODEL_PATH = './saved_models/MIGN_PubMed_best_r23456_256.pth'
DATASET = 'PubMed'
R = 6  # 训练时用的 r 值
HIDDEN = 256
DROPOUT = 0.6
NUM_RUNS = 100  # 测试次数
WARMUP = 50     # 预热次数（增加到50）
BATCH_SIZES = [1, 10, 100, 500, 1000, 2048]  # 不同批大小测试
# ==================================================


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_memory_usage(model, test_features, device):
    """测试GPU内存使用"""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # 模型占用的内存
        model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 推理时的峰值内存
        with torch.no_grad():
            _ = model(test_features)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        inference_memory = peak_memory - model_memory
        
        return model_memory, inference_memory, peak_memory
    else:
        return 0, 0, 0


def test_batch_inference(model, test_features, batch_sizes, device, num_runs=20):
    """测试不同批大小的推理速度"""
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(test_features):
            continue
            
        batch_data = test_features[:batch_size]
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(batch_data)
        
        # 测试
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(batch_data)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                times.append((end - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time * 1000  # samples/sec
        
        results[batch_size] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'per_sample_time': avg_time / batch_size
        }
    
    return results


def test_model_size(model_path):
    """测试模型文件大小"""
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return size_mb
    return 0


def main():
    """主测试函数"""
    print("="*80)
    print("增强版模型性能测试")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    user_features, user_labels, client_adj, label_num, _, _, \
    test_features, test_labels, val_features, val_labels, at_f, at_l = \
        deal_dataset(client_num=1, rate=1.0, r=R)
    
    feature_dim = len(user_features[0][0])
    print(f"   测试样本数量: {len(test_labels)}")
    print(f"   特征维度: {feature_dim}")
    
    # 2. 加载模型
    print("\n2. 加载模型...")
    model = GMLP(r=R, nfeat=feature_dim, nhid=HIDDEN, 
                 nclass=label_num, dropout=DROPOUT, drate=1.0)
    
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_features = test_features.to(device)
    model.eval()
    
    print(f"   使用设备: {device}")
    
    # 3. 模型参数统计
    print("\n3. 模型参数统计")
    total_params, trainable_params = count_parameters(model)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    print(f"   参数量(MB): {total_params * 4 / 1024 / 1024:.2f}")  # 假设float32
    
    # 4. 模型文件大小
    print("\n4. 模型文件大小")
    model_size = test_model_size(MODEL_PATH)
    print(f"   模型文件大小: {model_size:.2f} MB")
    
    # 5. 内存使用测试
    print("\n5. GPU内存使用测试")
    if device.type == 'cuda':
        model_mem, inference_mem, peak_mem = test_memory_usage(model, test_features, device)
        print(f"   模型内存占用: {model_mem:.2f} MB")
        print(f"   推理额外内存: {inference_mem:.2f} MB")
        print(f"   峰值内存占用: {peak_mem:.2f} MB")
    else:
        print("   使用CPU，跳过GPU内存测试")
    
    # 6. 准确率测试
    print("\n6. 测试准确率")
    with torch.no_grad():
        output = model(test_features)
        preds = output.max(1)[1]
        correct = preds.eq(torch.tensor(test_labels).to(device)).sum().item()
        accuracy = (correct / len(test_labels)) * 100
    print(f"   测试准确率: {accuracy:.2f}%")
    
    # 7. 整体推理速度测试（预热）
    print(f"\n7. 预热 ({WARMUP} 次)...")
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(test_features)
    
    # 8. 整体推理速度测试
    print(f"\n8. 整体推理速度测试 ({NUM_RUNS} 次)")
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
            times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = len(test_labels)/avg_time*1000
    
    print(f"   平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   最小时间: {min_time:.2f} ms")
    print(f"   最大时间: {max_time:.2f} ms")
    print(f"   整体吞吐量: {throughput:.0f} samples/sec")
    print(f"   单样本时间: {avg_time/len(test_labels)*1000:.3f} μs")
    
    # 9. 不同批大小测试
    print(f"\n9. 不同批大小推理测试")
    batch_results = test_batch_inference(model, test_features, BATCH_SIZES, device)
    
    print("\n   批大小  |  推理时间(ms)  |  吞吐量(samples/s)  |  单样本时间(μs)")
    print("   " + "-"*70)
    for batch_size, results in batch_results.items():
        print(f"   {batch_size:>6}  |  {results['avg_time']:>12.2f}  |  "
              f"{results['throughput']:>18.0f}  |  {results['per_sample_time']*1000:>15.2f}")
    
    # 10. 计算FLOPs（简化版）
    print("\n10. 计算复杂度估计")
    # 简单估计：对于MLP，FLOPs ≈ 2 * 输入维度 * 隐藏维度 * 层数
    # 这是一个粗略估计，实际FLOPs需要更详细的计算
    estimated_flops = 2 * feature_dim * HIDDEN * 2 * len(test_labels)  # 假设2层
    print(f"   估计FLOPs: {estimated_flops/1e9:.2f} GFLOPs")
    
    # 11. 结果汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"准确率:           {accuracy:.2f}%")
    print(f"模型参数量:       {total_params:,}")
    print(f"模型文件:         {model_size:.2f} MB")
    if device.type == 'cuda':
        print(f"GPU峰值内存:      {peak_mem:.2f} MB")
    print(f"平均推理时间:     {avg_time:.2f} ms ({len(test_labels)} samples)")
    print(f"吞吐量:           {throughput:.0f} samples/sec")
    print(f"单样本延迟:       {avg_time/len(test_labels)*1000:.3f} μs")
    
    # 最大批处理能力
    if batch_results:
        max_batch = max(batch_results.keys())
        max_throughput = max(r['throughput'] for r in batch_results.values())
        print(f"最大测试批:       {max_batch}")
        print(f"峰值吞吐量:       {max_throughput:.0f} samples/sec")
    print("="*80)
    
    # 保存结果
    results = {
        'accuracy': accuracy,
        'total_params': total_params,
        'model_size_mb': model_size,
        'avg_inference_time_ms': avg_time,
        'throughput_samples_per_sec': throughput,
        'batch_results': batch_results
    }
    
    return results


if __name__ == '__main__':
    results = main()