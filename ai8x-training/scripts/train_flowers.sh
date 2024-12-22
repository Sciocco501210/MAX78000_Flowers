#!/bin/sh
python train.py --epochs 10 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-flowers.yaml --qat-policy policies/qat_policy_flowers.yaml --model ai85flowernet --dataset FlowerClassification --data data/FlowerClassification --confusion --param-hist --embedding --enable-tensorboard --device MAX78000 "$@"
