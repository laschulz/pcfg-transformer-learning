#!/bin/bash

SUPERGRAMMAR="OverlappingSubgrammar_plus"
SUBGRAMMAR_TRAIN="L1_subgrammar"
SUBGRAMMAR_ANALYSIS="L1"
DATASET_SIZE=300
MODEL="OneLayer"
CONTINUE_FROM=41
NUM_EPOCHS=70

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L0 

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol L0 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

# ------- Training -------

python train.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --model $MODEL

python train.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE\
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_TRAIN}/${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}/${MODEL}/new/seed_42/epoch_${CONTINUE_FROM}.pt" \
    --num_epochs $NUM_EPOCHS

python train.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --continue_from $CONTINUE_FROM \
    --num_epochs $NUM_EPOCHS 


# ------- Analysis -------

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L0 \
    --to_epoch 150 \
    --subgrammar $SUPERGRAMMAR \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L0_direct \
    --to_epoch 150 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L1 \
    --to_epoch 150 \
    --subgrammar $SUBGRAMMAR_ANALYSIS \
    --train_type continued