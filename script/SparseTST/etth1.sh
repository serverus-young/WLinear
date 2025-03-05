#!/bin/bash
for pred_len in 96
do
  python test.py \
      --data_path './dataset/ETT-small/ETTh1.csv' \
      --data ETTh \
      --Data ETTh1 \
      --model WT2 \
      --seq_len 360 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003 \
      --patience 10
done
