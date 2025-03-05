#!/bin/bash
for pred_len in 96 192 360 720
do
for seq_len in 360 720
do
for lr in 0.0005 0.001 0.003
do
for d_model in 32 64 128 256
do
for batch_size in 16 32 64
do
  python test.py \
      --data_path './dataset/traffic.csv' \
      --data custom \
      --Data traffic \
      --model WFT \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --save_result 1 \
      --lr $lr \
      --patience 5 \
      --batch_size $batch_size \
      --e_in 862 \
      --d_model $d_model \
      --train_mode 1 \
      --level 2
done
done
done
done
done