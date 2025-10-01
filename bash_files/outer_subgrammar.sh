#!/bin/bash

SUPERGRAMMAR="Grammar_with_simple_possibilities"
SUPERGRAMMAR_SYMBOL="START"
SUBGRAMMAR_1_ANALYSIS="Grammar_with_only_simple_possibilities"
DATASET_SIZE=50000
MODEL="TwoLayer"
TO_EPOCH=2

cd ../src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol START 

python train.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --num_epochs $TO_EPOCH

#-------ANALYSIS--------

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUPERGRAMMAR_SYMBOL \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUPERGRAMMAR \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal START_simple \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUBGRAMMAR_1_ANALYSIS \
    --train_type new