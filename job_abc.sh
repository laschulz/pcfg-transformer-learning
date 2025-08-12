#!/bin/bash

SUPERGRAMMAR="ABC_grammar"
SUBGRAMMAR_A_ANALYSIS="L1"
SUBGRAMMAR_B_TRAIN="L1_3c_subgrammar"
SUBGRAMMAR_B_ANALYSIS="L1_3c"
SUBGRAMMAR_C_ANALYSIS="L1_2b"
DATASET_SIZE=300
MODEL="OneLayer"
CONTINUE_FROM_1=40

cd src
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L0 

# python generate_pcfg.py --grammar $SUBGRAMMAR_B_TRAIN --dataset_size $DATASET_SIZE --start_symbol L0 \
#     --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

# # -------TRAINING--------
# python train.py --grammar $SUBGRAMMAR_B_TRAIN --dataset_size $DATASET_SIZE --model $MODEL --num_epochs 40

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_size $DATASET_SIZE \
#     --model $MODEL \
#     --continue_training \
#     --checkpoint_path "${SUBGRAMMAR_B_TRAIN}/${SUBGRAMMAR_B_TRAIN}_${DATASET_SIZE}/${MODEL}/new/seed_42/epoch_${CONTINUE_FROM_1}.pt"

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_size $DATASET_SIZE \
#     --model $MODEL \
#     --continue_from $CONTINUE_FROM_1

# # -------ANALYSIS--------

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_size $DATASET_SIZE \
#     --model $MODEL \
#     --nonTerminal L0 \
#     --to_epoch 150 \
#     --subgrammar $SUPERGRAMMAR \
#     --train_type continued

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_size $DATASET_SIZE \
#     --model $MODEL \
#     --nonTerminal L0_direct \
#     --to_epoch 150 \
#     --subgrammar $SUPERGRAMMAR \
#     --train_type new

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_size $DATASET_SIZE \
#     --model $MODEL \
#     --nonTerminal L1 \
#     --to_epoch 150 \
#     --subgrammar $SUBGRAMMAR_A_ANALYSIS \
#     --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1_3c \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_B_ANALYSIS \
    --train_type continued \

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1_2b \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_C_ANALYSIS \
    --train_type continued