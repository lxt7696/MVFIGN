#!/bin/bash
# run_ablation_cora_corrected.sh - 修正版消融实验脚本

python -u v_main_mpche.py --dataSet Cora --round1 50 --epochs2 300 2>&1 | tee -a table10_mvfign_cora_mpche_0.9.txt
python -u v_main_mpche.py --dataSet Citeseer --round1 50 --epochs2 300 2>&1 | tee -a table10_mvfign_cora_mpche.txt
python -u v_main_mpche.py --dataSet Pubmed --round1 50 --epochs2 300 2>&1 | tee -a table10_mvfign_cora_mpche.txt

python -u v_main_mpche.py --dataSet Pubmed --round1 50 --epochs2 300 2>&1 | tee -a table10_mvfign_pub_mpche_0.9.txt

python -u v_main_mpche_1.py --round1 50 --epochs2 300 2>&1 | tee -a 1__mvfign_core_0.9.txt
