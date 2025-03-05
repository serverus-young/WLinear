#!/bin/bash
for pred_len in 96 192 336 720
do
    python test.py \
      --data_path './dataset/CoalCH4.csv' \
      --data custom \
      --Data coal \
      --model TimesNets \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --c_in  8 \
      --lr 0.0001 \
      --patience 5 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32
      python test.py \
      --data_path './dataset/CoalCH4.csv' \
      --data custom \
      --Data coal \
      --model PatchTST \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0001 \
      --c_in 8 \
      --patience 5 \
      --batch_size 16 \
      --d_model 512 \
      --d_ff 2048
      python test.py \
      --data_path './dataset/CoalCH4.csv' \
      --data custom \
      --Data coal \
      --model WFT \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --c_in 8 \
      --lr 0.003 \
      --patience 5 \
      --batch_size 16
      python test.py \
      --data_path './dataset/CoalCH4.csv' \
      --data custom \
      --Data coal \
      --model FITS \
      --seq_len 720 \
      --pred_len $pred_len \
      --save_result 1 \
      --lr 0.0005 \
      --c_in  8 \
      --patience 5 \
      --batch_size 16 \
      --H_order 6
done
