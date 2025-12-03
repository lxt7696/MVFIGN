import torch
from v_deal import deal_dataset

client_num = 2
rate = 1.0
r = 3

print("Loading dataset...")
user_features, user_labels, client_adj, label_num, train_features, train_labels, \
    test_features, test_labels, at_f, at_l = deal_dataset(client_num, rate, r)

print("\n" + "="*60)
print("DIMENSION ANALYSIS")
print("="*60)

# user_features 维度
print(f"\n1. user_features structure:")
print(f"   - Number of clients: {len(user_features)}")
for i in range(len(user_features)):
    print(f"   - Client {i}: {len(user_features[i])} samples")
    if len(user_features[i]) > 0:
        print(f"     First sample shape: {user_features[i][0].shape}")
        # 如果是多个样本，检查堆叠后的形状
        stacked = torch.stack(user_features[i])
        print(f"     Stacked shape: {stacked.shape}")  # (num_samples, feature_dim)

# at_f 维度
print(f"\n2. at_f (training features for all clients):")
print(f"   Shape: {at_f.shape}")  # 应该是 (num_clients, num_samples, feature_dim)

# test_features 维度
print(f"\n3. test_features:")
print(f"   Shape: {test_features.shape}")

# 推断实际需要的 m 和 n
print(f"\n" + "="*60)
print("REQUIRED MPC DIMENSIONS")
print("="*60)

if len(user_features[0]) > 0:
    # 对于单个样本
    n = user_features[0][0].shape[0]  # 特征维度
    print(f"n (feature dimension) = {n}")
    
    # 对于批次
    if hasattr(client_adj[0], 'shape'):
        print(f"Adjacency matrix shape: {client_adj[0].shape}")
        max_m = client_adj[0].shape[0]  # 最大样本数
        print(f"max_m (max samples in adjacency) = {max_m}")
    
    # at_f 的批次大小
    print(f"at_f batch size (num_samples) = {at_f.shape[1]}")

print("="*60)