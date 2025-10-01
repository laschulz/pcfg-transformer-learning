#!/bin/bash

SUPERGRAMMAR="ABC_grammar"
SUPERGRAMMAR_SYMBOL="L0"
SUBGRAMMAR_SYMBOL="L1b"
SUBGRAMMAR_TRAIN="L1b_subgrammar"
DATASET_SIZE=50000
MODEL="TwoLayer_SMALL"
PRETRAIN_EPOCHS=3
TO_EPOCH=10

cd src

python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL 

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

# ------- Training -------
python train.py --grammar $SUBGRAMMAR_TRAIN --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" --model $MODEL --num_epochs $PRETRAIN_EPOCHS --seed 42

python train.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --continue_training \
    --checkpoint_path "${SUBGRAMMAR_TRAIN}/${SUBGRAMMAR_TRAIN}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/new/seed_42/epoch_${PRETRAIN_EPOCHS}_0.pt" \
    --num_epochs $TO_EPOCH \
    --continue_from $PRETRAIN_EPOCHS

# -------ANALYSIS--------

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_SYMBOL\
    --to_epoch 10 \
    --subgrammar $SUBGRAMMAR_SYMBOL \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUPERGRAMMAR_SYMBOL \
    --to_epoch 10 \
    --subgrammar $SUPERGRAMMAR \
    --train_type continued