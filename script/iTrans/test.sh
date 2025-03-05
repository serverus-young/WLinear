#!/bin/bash
for pred_len in 96 192 336 720
do
  python test.py \
      --data_path './dataset/ETT-small/ETTh1.csv' \
      --data ETTh \
      --Data ETTh1 \
      --model WFT \
      --seq_len 720\
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003
  python test.py \
      --data_path './dataset/weather.csv' \
      --data custom \
      --Data Weather \
      --model WFT \
      --seq_len 720\
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003
  python test.py \
      --data_path './dataset/electricity.csv' \
      --data custom \
      --Data electricity \
      --model WFT \
      --seq_len 720\
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003
  python test.py \
      --data_path './dataset/traffic.csv' \
      --data custom \
      --Data Traffic \
      --model WFT \
      --seq_len 720\
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003
  python test.py \
      --data_path './dataset/CoalCH4.csv' \
      --data custom \
      --Data Coal \
      --model WFT \
      --seq_len 720\
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003
done
