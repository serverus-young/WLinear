#!/bin/bash
for pred_len in 96 192 360 720
do
for seq_len in 360 720
do
for d_model in 64 128 256 512
do
for batch_size in  32 64 128
do
  python test.py \
      --data_path './dataset/weather.csv' \
      --data custom \
      --Data weather \
      --model WFT \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003 \
      --patience 5 \
      --batch_size $batch_size \
      --e_in 21 \
      --d_model $d_model \
      --train_mode 1 \
      --level 2 \
      --device 0 \
      --individual
done
done
done
done