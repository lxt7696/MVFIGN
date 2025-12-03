#!/bin/bash
# 第二层级：组件消融分析

# 实验3：多视角模块消融
python mul_main.py --dataSet Cora --client_num 2 --cm 0.5 --d 1  #  完整MVFIGN
python mul_main.py --dataSet Cora --client_num 2 --cm 0.5 --d 2  # 移除多视角融合

# 实验4：正则化组件分析
python mul_main.py --dataSet Cora --client_num 2 --cm 0.0  # 无丢弃
python mul_main.py --dataSet Cora --client_num 2 --cm 0.3  # 中等丢弃
python mul_main.py --dataSet Cora --client_num 2 --cm 0.6  # 高丢弃