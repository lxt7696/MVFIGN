#废
#!/bin/bash
# run_fair_ablation.sh

echo "Fair Ablation Study: MIGN vs MVFIGN with SAME parameters"

# 统一参数设置
ROUND1=10
EPOCHS2=250  # 取中间值
CM=1.0       # 使用默认值
HIDDEN=384   # 取中间值

# ============ MIGN (Centralized) ============
echo -e "\n[1/6] MIGN on Cora..."
python mul_main.py --dataSet Cora --r 6 --round1 $ROUND1 --round2 $EPOCHS2 --cm $CM --hidden $HIDDEN

echo -e "\n[2/6] MIGN on Citeseer..."
python mul_main.py --dataSet Citeseer --r 6 --round1 $ROUND1 --round2 $EPOCHS2 --cm $CM --hidden $HIDDEN

echo -e "\n[3/6] MIGN on Pubmed..."
python mul_main.py --dataSet PubMed --r 6 --round1 $ROUND1 --round2 $EPOCHS2 --cm $CM --hidden $HIDDEN

# ============ MVFIGN (Federated) ============
echo -e "\n[4/6] MVFIGN on Cora..."
python v_main.py --dataSet Cora --client_num 2 --round1 $ROUND1 --epochs2 $EPOCHS2 --cm $CM --hidden $HIDDEN

echo -e "\n[5/6] MVFIGN on Citeseer..."
python v_main.py --dataSet Citeseer --client_num 2 --round1 $ROUND1 --epochs2 $EPOCHS2 --cm $CM --hidden $HIDDEN

echo -e "\n[6/6] MVFIGN on Pubmed..."
python v_main.py --dataSet PubMed --client_num 2 -round1 $ROUND1 --epochs2 $EPOCHS2 --cm $CM --hidden $HIDDEN