#!/bin/bash
for pred_len in 96 192 336 720
do
  python test.py \
      --data_path './dataset/ETT-small/ETTh1.csv' \
      --data ETTh \
      --Data ETTh1 \
      --model WFT \
      --seq_len $pred_len\
      --pred_len 96 \
      --save_result 1 \
      --lr 0.003
done
