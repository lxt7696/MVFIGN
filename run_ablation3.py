#!/bin/bash
# 第三层级：联邦学习参数敏感性

# 实验5：客户端数量影响
for client_num in 1 2 4 6; do
    python mul_main.py --dataSet Cora --client_num $client_num --cm 0.5
done

# 实验6：通信轮次影响
for round1 in 50 100 200; do
    python mul_main.py --dataSet Cora --client_num 2 --cm 0.5 --round1 $round1
done