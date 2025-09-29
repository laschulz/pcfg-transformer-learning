#!/bin/bash
SUPERGRAMMAR="easyLingo2"
DATASET_SIZE=50000 #50k
SUPERGRAMMAR_SYMBOL="L4"
MODEL="TwoLayer_31"

cd src

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --model $MODEL \
        --nonTerminal $SUPERGRAMMAR_SYMBOL\
        --to_epoch 5 \
        --subgrammar $SUPERGRAMMAR \
        --train_type new \
        --seed 1 2 3 4 5 42 \
        --plot_only