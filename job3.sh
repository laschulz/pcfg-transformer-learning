#!/bin/bash

SUPERGRAMMAR="ABC_grammar_basic"
SUBGRAMMAR_A_ANALYSIS="L2_3"
SUBGRAMMAR_B_TRAIN="L2_3b_subgrammar"
SUBGRAMMAR_B_ANALYSIS="L2_3b"
SUBGRAMMAR_C_ANALYSIS="L2_3c"
DATASET_SIZE=300
MODEL="FourLayer"
CONTINUE_FROM_1=21

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L1 

python generate_pcfg.py --grammar $SUBGRAMMAR_B_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python generate_pcfg.py --grammar $SUBGRAMMAR_B_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

# -------TRAINING--------
python train.py --grammar $SUBGRAMMAR_B_TRAIN --dataset_size $DATASET_SIZE --model $MODEL --num_epochs 30

python train.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_B_TRAIN}/${SUBGRAMMAR_B_TRAIN}_${DATASET_SIZE}/${MODEL}/new/seed_42/epoch_${CONTINUE_FROM_1}.pt"

python train.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --continue_from $CONTINUE_FROM_1

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
    --nonTerminal L2_3 \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_A_ANALYSIS \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L2_3b \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_B_ANALYSIS \
    --train_type continued \
    --plot_only

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L2_3c \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_C_ANALYSIS \
    --train_type continued