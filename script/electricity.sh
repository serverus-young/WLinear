#!/bin/bash

for pred_len in 336 720
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
      --patience 10 \
      --batch_size 16 \
      --e_in 321

done
