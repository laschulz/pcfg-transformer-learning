#!/bin/bash -l

SUPERGRAMMAR="KL_decomposition_example1"
SUPERGRAMMAR_SYMBOL="L1"
SUBGRAMMAR_TRAIN="L2_1"
SUBGRAMMAR="L2_1_subgrammar"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer"
PRETRAIN_EPOCHS=2
EPOCHS=3

cd ../src


# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# evaluating without pretraining
# TODO: CHANGE THE 5

# python train.py \
#     --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs 3 \
#     --seed 42

# python attention_heads.py --model $MODEL \
#                             --grammar $SUPERGRAMMAR \
#                             --checkpoint_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/new/seed_42" \
#                             --base_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#                             --subgrammar $SUBGRAMMAR


# evaluating with pretraining
# python generate_pcfg.py --grammar $SUBGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL \
#     --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

# python train.py --grammar $SUBGRAMMAR --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" --model $MODEL --num_epochs $PRETRAIN_EPOCHS --seed 42

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --checkpoint_path "${SUBGRAMMAR}/${SUBGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/new/seed_42/epoch_${PRETRAIN_EPOCHS}_0.pt" \
#     --num_epochs $EPOCHS \
#     --continue_from $PRETRAIN_EPOCHS

python attention_heads.py --model $MODEL \
                            --grammar $SUPERGRAMMAR \
                            --checkpoint_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/continued/seed_42" \
                            --base_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
                            --subgrammar $SUBGRAMMAR