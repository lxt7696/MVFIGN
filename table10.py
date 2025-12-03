对于集中式版本（mul_main.py）：​​
数据集: Cora, Citeseer, Pubmed
命令模板: python mul_main.py --client_num 1 --dataSet [dataset] --round1 300 --lr 0.001 --r 6 --cuda

​对于联邦学习版本（v_main.py）：​​
数据集: Cora, Citeseer, Pubmed
分割比例: 0.9, 0.8, 0.7, 0.5
命令模板: python v_main.py --client_num 2 --dataSet [dataset] --rate 1.0 --round1 10 --round2 300 --lr 0.01 --r 3 --edge_split_ratio [ratio] --cuda

python -u mul_main.py --client_num 1 --dataSet Cora --round1 10 --epochs2 300 --lr 0.001 --r 6 2>&1 | tee -a table10_mign_co.txt
python -u mul_main.py --client_num 1 --dataSet Citeseer --round1 10 --epochs2 300 --lr 0.001 --r 6 2>&1 | tee -a table_mvfign_ci.txt
python -u mul_main.py --client_num 1 --dataSet Pubmed --round1 10 --epochs2 300 --lr 0.001 --r 6 2>&1 | tee -a table_mvfign_p.txt

python -u v_main.py --client_num 2 --dataSet Cora --round1 10 --epochs2 300 --lr 0.01 --r 3 2>&1 | tee -a table10_mvfign_co_5.txt
python -u v_main.py --client_num 2 --dataSet Citeseer --round1 10 --epochs2 300 --lr 0.01 --r 3 2>&1 | tee -a table10_mvfign_ci_5.txt
python -u v_main.py --client_num 2 --dataSet Pubmed --round1 10 --epochs2 300 --lr 0.01 --r 3 2>&1 | tee -a table10_mvfign_p_5.txt