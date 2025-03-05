#!/bin/bash
for pred_len in 96
do
for m in 1 2
do
for seed in 1218 1024 1003
do
  python test.py \
      --data_path './dataset/ETT-small/ETTh1.csv' \
      --data ETTh \
      --Data ETTh1 \
      --model WFT \
      --seq_len 360 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.003 \
      --batch_size 128 \
      --d_model 64 \
      --patience 10 \
      --train_mode $m \
      --seed $seed
done
done
done

