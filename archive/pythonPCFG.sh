#!/bin/bash

SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
SUBGRAMMAR_SYMBOL="compound_stmt"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer_SMALL"
CONTINUE_FROM=5
NUM_EPOCHS=40

cd ../src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUBGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

# # ------- Training -------
for SEED in 1 2 3; do
    python train.py --grammar $SUPERGRAMMAR --dataset_name "${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}" \
        --model $MODEL --num_epochs $CONTINUE_FROM --seed $SEED

    python train.py --grammar $SUPERGRAMMAR \
        --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --model $MODEL \
        --continue_training \
        --checkpoint_path "${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}/${MODEL}/new/seed_${SEED}/epoch_${CONTINUE_FROM}_0.pt" \
        --num_epochs $NUM_EPOCHS \
        --continue_from $CONTINUE_FROM \
        --seed $SEED

    python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --model $MODEL \
        --nonTerminal $SUPERGRAMMAR_SYMBOL \
        --to_epoch 50 \
        --subgrammar $SUPERGRAMMAR \
        --train_type continued \
        --seed $SEED
done

python train.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --num_epochs $((CONTINUE_FROM + NUM_EPOCHS)) \
    --seed 5


python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal "${SUPERGRAMMAR_SYMBOL}_direct" \
    --to_epoch 50 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new \
    --seed 5
    
