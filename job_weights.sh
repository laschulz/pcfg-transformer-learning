#!/bin/bash

SUPERGRAMMAR="OverlappingSubgrammar_plus"
SUBGRAMMAR_TRAIN="L1_subgrammar"
SUBGRAMMAR_ANALYSIS="L1"
DATASET_SIZE=300
MODEL="OneLayer"
CONTINUE_FROM=50

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L0 

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol L0 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python weight_space.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL \
    --subgrammar $SUBGRAMMAR_TRAIN --num_epochs_direct 50 --num_epochs_pretrain 40