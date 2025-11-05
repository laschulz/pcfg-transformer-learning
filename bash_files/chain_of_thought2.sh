#!/bin/bash

SUPERGRAMMAR="ABC_grammar"
SUPERGRAMMAR_SYMBOL="L0"
SUBGRAMMAR_SYMBOL1="L1b"
SUBGRAMMAR_TRAIN1="L1b_subgrammar"
SUBGRAMMAR_SYMBOL2="L1c"
SUBGRAMMAR_TRAIN2="L1c_subgrammar"
SUBGRAMMAR_SYMBOL3="L1a"
SUBGRAMMAR_TRAIN3="L1a_subgrammar"
DATASET_SIZE=50000
MODEL="TwoLayer_SMALL"
PRETRAIN_EPOCHS=3
TO_EPOCH=10

cd ../src

python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL 

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN1 --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN2 --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN3 --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250


# ------- Training -------
# 1. Pretrain on first subgrammar
python train.py \
    --grammar $SUBGRAMMAR_TRAIN1 \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --num_epochs $PRETRAIN_EPOCHS \
    --seed 42

# 2. Continue on second subgrammar, starting from checkpoint of step 1
python train.py \
    --grammar $SUBGRAMMAR_TRAIN2 \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --checkpoint_path "${SUBGRAMMAR_TRAIN1}/${SUBGRAMMAR_TRAIN1}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/new/seed_42/epoch_${PRETRAIN_EPOCHS}_0.pt" \
    --num_epochs $PRETRAIN_EPOCHS \
    --continue_from $PRETRAIN_EPOCHS \
    --seed 42

# Train on first subgrammar again
python train.py \
    --grammar $SUBGRAMMAR_TRAIN1 \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --checkpoint_path "${SUBGRAMMAR_TRAIN2}/${SUBGRAMMAR_TRAIN2}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/continued/seed_42/epoch_$((2 * PRETRAIN_EPOCHS))_0.pt" \
    --num_epochs $PRETRAIN_EPOCHS \
    --continue_from $((2 * PRETRAIN_EPOCHS)) \
    --seed 42

# Train on new subgramar
python train.py \
    --grammar $SUBGRAMMAR_TRAIN3 \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --checkpoint_path "${SUBGRAMMAR_TRAIN1}/${SUBGRAMMAR_TRAIN1}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/continued/seed_42/epoch_$((3 * PRETRAIN_EPOCHS))_0.pt" \
    --num_epochs $PRETRAIN_EPOCHS \
    --continue_from $((3 * PRETRAIN_EPOCHS)) \
    --seed 42

# 3. Train on the supergrammar, starting from checkpoint of step 2
python train.py \
    --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --checkpoint_path "${SUBGRAMMAR_TRAIN3}/${SUBGRAMMAR_TRAIN3}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/continued/seed_42/epoch_$((4 * PRETRAIN_EPOCHS))_0.pt" \
    --num_epochs $TO_EPOCH \
    --continue_from $((4 * PRETRAIN_EPOCHS)) \
    --seed 42


# -------ANALYSIS--------

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUPERGRAMMAR_SYMBOL \
    --to_epoch 30 \
    --subgrammar $SUPERGRAMMAR \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_SYMBOL1\
    --to_epoch 30 \
    --subgrammar $SUBGRAMMAR_SYMBOL1 \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_SYMBOL2 \
    --to_epoch 30 \
    --subgrammar $SUBGRAMMAR_SYMBOL2 \
    --train_type continued

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUBGRAMMAR_SYMBOL3 \
    --to_epoch 30 \
    --subgrammar $SUBGRAMMAR_SYMBOL3 \
    --train_type continued