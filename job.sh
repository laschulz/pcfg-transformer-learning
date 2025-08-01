#!/bin/bash

SUPERGRAMMAR="O3_Combined"
SUBGRAMMAR_TRAIN="L2"
SUBGRAMMAR_ANALYSIS="L2"
DATASET_SIZE=300
MODEL="OneLayer"
CONTINUE_FROM=10

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L1 

python generate_pcfg.py --grammar "{$SUBGRAMMAR_TRAIN}" --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python train.py --pcfg "{$SUBGRAMMAR_TRAIN}" --dataset "${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}" --model $MODEL

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "../data/${SUBGRAMMAR_TRAIN}/${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}/${MODEL}/continued/epoch_${CONTINUE_FROM}.pt"

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_from $CONTINUE_FROM

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1 \
    --to_epoch 100
    --subgrammar $SUPERGRAMMAR

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1_direct \
    --to_epoch 100
    --subgrammar $SUPERGRAMMAR

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L2 \
    --to_epoch 100
    --subgrammar $SUBGRAMMAR_ANALYSIS