#!/bin/bash

SUPERGRAMMAR="O3_Combined"
SUBGRAMMAR_1_TRAIN="L2_verysimple_subgrammar"
SUBGRAMMAR_1_ANALYSIS="L2_verysimple"
SUBGRAMMAR_2_TRAIN="L2_3_subgrammar"
SUBGRAMMAR_2_ANALYSIS="L2_3"
DATASET_SIZE=300
MODEL="FourLayer"
CONTINUE_FROM_1=21
CONTINUE_FROM_2=42

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L1 

python generate_pcfg.py --grammar $SUBGRAMMAR_1_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python generate_pcfg.py --grammar $SUBGRAMMAR_2_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

# -------TRAINING--------
python train.py --pcfg $SUBGRAMMAR_1_TRAIN --dataset "${SUBGRAMMAR_1_TRAIN}_${DATASET_SIZE}" --model $MODEL --num_epochs 30

python train.py --pcfg $SUBGRAMMAR_2_TRAIN \
    --dataset "${SUBGRAMMAR_2_TRAIN}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_1_TRAIN}/${SUBGRAMMAR_1_TRAIN}_${DATASET_SIZE}/${MODEL}/new/epoch_${CONTINUE_FROM_1}.pt" \
    --num_epochs 30

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_2_TRAIN}/${SUBGRAMMAR_2_TRAIN}_${DATASET_SIZE}/${MODEL}/continued/epoch_${CONTINUE_FROM_2}.pt"

python train.py --pcfg $SUPERGRAMMAR \
    --dataset "${SUPERGRAMMAR}_${DATASET_SIZE}" \
    --model $MODEL \
    --continue_from $CONTINUE_FROM_2

# -------ANALYSIS--------

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
    --subgrammar $SUBGRAMMAR_1_ANALYSIS \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L2_3 \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_2_ANALYSIS \
    --train_type continued