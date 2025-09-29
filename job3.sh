#!/bin/bash

SUPERGRAMMAR="KL_decomposition_example2"
SUPERGRAMMAR_SYMBOL="L1"
SUBGRAMMAR_1_ANALYSIS="L2_1"
SUBGRAMMAR_2_ANALYSIS="L2_2"
SUBGRAMMAR_3_ANALYSIS="L2_3"
DATASET_SIZE=50000
MODEL="TwoLayer"
TO_EPOCH=2

cd src
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol START 

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs $TO_EPOCH

# # #-------ANALYSIS--------


# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --nonTerminal $SUBGRAMMAR_1_ANALYSIS\
#     --to_epoch $TO_EPOCH \
#     --subgrammar $SUBGRAMMAR_1_ANALYSIS \
#     --train_type new

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --nonTerminal $SUBGRAMMAR_2_ANALYSIS\
#     --to_epoch $TO_EPOCH \
#     --subgrammar $SUBGRAMMAR_2_ANALYSIS \
#     --train_type new

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --nonTerminal $SUBGRAMMAR_2_ANALYSIS\
#     --to_epoch $TO_EPOCH \
#     --subgrammar $SUBGRAMMAR_2_ANALYSIS \
#     --train_type new \
#     --plot_only

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal overhead \
    --to_epoch $TO_EPOCH \
    --subgrammar overhead \
    --train_type new \
    --plot_only
