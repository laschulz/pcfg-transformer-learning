#!/bin/bash

SUPERGRAMMAR="Deeper_Recursion"
SUPERGRAMMAR_SYMBOL="L0"
SUBGRAMMAR_1_ANALYSIS="L1"
SUBGRAMMAR_2_ANALYSIS="L2"
SUBGRAMMAR_3_ANALYSIS="L3"
SUBGRAMMAR_4_ANALYSIS="L4"
DATASET_SIZE=50000
MODEL="TwoLayer"
TO_EPOCH=2

cd ../src
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL 

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs $TO_EPOCH

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
    --nonTerminal $SUBGRAMMAR_1_ANALYSIS \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUBGRAMMAR_1_ANALYSIS \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_2_ANALYSIS \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUBGRAMMAR_2_ANALYSIS \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_3_ANALYSIS \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUBGRAMMAR_3_ANALYSIS \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_4_ANALYSIS \
    --to_epoch $TO_EPOCH \
    --subgrammar $SUBGRAMMAR_4_ANALYSIS \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal L1 \
    --to_epoch $TO_EPOCH \
    --subgrammar overhead \
    --train_type new