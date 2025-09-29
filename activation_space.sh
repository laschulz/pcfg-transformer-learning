SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer_SMALL"
SEED=21 # this is the best seed

cd src

python activation_space.py \
    --model $MODEL \
    --base_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --seeds $SEED \
    --epoch 65 \
    --train_type new \
    --plot

# python activation_space.py \
#     --model $MODEL \
#     --base_dir "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --seeds $SEED \
#     --epoch 55 \
#     --train_type continued