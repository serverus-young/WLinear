#!/bin/bash

for pred_len in 96 192 336 720
do
  python test.py \
      --data_path './dataset/traffic.csv' \
      --data custom \
      --Data traffic \
      --model WFT \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0030 \
      --patience 10 \
      --batch_size 16 \
      --e_in 862
   python test.py \
      --data_path './dataset/traffic.csv' \
      --data custom \
      --Data traffic \
      --model FITS \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0005 \
      --patience 10 \
      --batch_size 64 \
      --H_order 10 \
      --e_in 862
done
