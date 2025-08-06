#!/bin/bash

SUPERGRAMMAR="O3_Combined"
SUBGRAMMAR_TRAIN="L2_verysimple_subgrammar"
SUBGRAMMAR_ANALYSIS="L2_verysimple"
DATASET_SIZE=300
MODEL="TwoLayer"
CONTINUE_FROM=46

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L1 

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol L1 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python weight_space.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL \
    --subgrammar $SUBGRAMMAR_TRAIN --num_epochs_direct 50 --num_epochs_pretrain 40