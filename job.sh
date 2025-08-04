#!/bin/bash

SUPERGRAMMAR="O3_Combined"
SUBGRAMMAR_TRAIN="L2_verysimple_subgrammar"
SUBGRAMMAR_ANALYSIS="L2_verysimple"
DATASET_SIZE=300
MODEL="OneLayer"
CONTINUE_FROM=46

cd src
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L1 

# python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
#     --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python train.py --pcfg $SUBGRAMMAR_TRAIN --dataset "${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}" --model $MODEL

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_TRAIN}/${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}/${MODEL}/new/epoch_${CONTINUE_FROM}.pt"

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_from $CONTINUE_FROM

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1 \
    --to_epoch 150 \
    --subgrammar $SUPERGRAMMAR \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1_direct \
    --to_epoch 150 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L2 \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_ANALYSIS \
    --train_type continued