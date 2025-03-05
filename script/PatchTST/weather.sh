#!/bin/bash

for pred_len in 96 192 336 720
do
  python test.py \
      --data_path './dataset/weather.csv' \
      --data custom \
      --Data weather \
      --model WFT \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0030 \
      --patience 5 \
      --batch_size 16 \
      --e_in 21
   python test.py \
      --data_path './dataset/weather.csv' \
      --data custom \
      --Data weather \
      --model FITS \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0005 \
      --patience 5 \
      --batch_size 64 \
      --H_order 12 \
      --base_T 144 \
      --e_in 21
done
