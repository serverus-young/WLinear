#!/bin/bash

for pred_len in 96
do
  python test.py \
      --data_path './dataset/electricity.csv' \
      --data custom \
      --Data electricity \
      --model WFT \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0030 \
      --patience 5 \
      --batch_size 16 \
      --e_in 321
  python test.py \
      --data_path './dataset/electricity.csv' \
      --data custom \
      --Data coal \
      --model FITS \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0005 \
      --patience 5 \
      --batch_size 16 \
      --H_order 10
done
