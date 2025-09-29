SUPERGRAMMAR="easyLingo2"
SUPERGRAMMAR_SYMBOL="L4"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer_31"
SEED=5


cd src

# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs 5 \
#     --seed 6

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs 5 \
#     --seed 7


for i in 3; do
    python logit_comparison.py \
        --model $MODEL \
        --path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --case $i \
        --seeds 1 2 4 5 42
done

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --nonTerminal $SUPERGRAMMAR_SYMBOL \
#     --to_epoch 5 \
#     --subgrammar $SUPERGRAMMAR \
#     --train_type new