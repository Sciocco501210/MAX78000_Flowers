#!/bin/sh
#python train.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-catsdogs-qat8-q.pth.tar -8 --device MAX78000 "$@"
#python train.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-training/logs/2024.11.05-033329/qat_best-quantized.pth.tar -8 --save-sample 1 --device MAX78000 "$@"
MODEL="ai85cdnet"
DATASET="cats_vs_dogs"
QUANTIZED_MODEL="../ai8x-training/logs/2024.11.05-033329/qat_best-quantized.pth.tar"


#python ai8xize.py --test-dir $TARGET --prefix cats-dogs --checkpoint-file trained/ai85-catsdogs-qat8-q.pth.tar --config-file networks/cats-dogs-hwc.yaml --fifo --softmax $COMMON_ARGS "$@"

python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"