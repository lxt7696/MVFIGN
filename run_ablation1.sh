#废
#!/bin/bash
# 第一层级：核心模型对比
#./run_ablation1.sh   # 一次运行所有实验

mkdir -p ablation_results/layer1

# 实验1：基础对比（固定参数）
python v_main.py --dataSet Cora --client_num 2 --cm 0.5 --round1 50 --epochs2 200 --lr 0.01 > results/v_main_baseline.log 2>&1
python mul_main.py --dataSet Cora --client_num 2 --cm 0.5 --round1 50 --epochs2 200 --lr 0.01 > results/mul_main_baseline.log 2>&1

# 实验2：多配置验证
for client in 2 4; do
    for cm in 0.3 0.6; do
        python v_main.py --dataSet Cora --client_num $client --cm $cm --round1 50 --epochs2 200 > ablation_results/layer1/v_main_c${client}_cm${cm}.log 2>&1
        python mul_main.py --dataSet Cora --client_num $client --cm $cm --round1 50 --epochs2 200 > ablation_results/layer1/mul_main_c${client}_cm${cm}.log 2>&1
    done
done

echo "结果保存在: ablation_results/layer1/"