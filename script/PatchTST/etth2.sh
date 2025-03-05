#!/bin/bash


for pred_len in 96 192 336 720
do
  python test.py \
      --data_path './dataset/ETT-small/ETTh2.csv' \
      --data ETTh \
      --Data ETTh2 \
      --model WFT \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0030 \
      --patience 10 \
      --batch_size 16
   python test.py \
      --data_path './dataset/ETT-small/ETTh2.csv' \
      --data ETTh \
      --Data ETTh2 \
      --model FITS \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0005 \
      --patience 10 \
      --batch_size 64 \
      --H_order 6
done