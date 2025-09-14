SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer_LARGER"

cd src

for i in {20..29}; do
    python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --model $MODEL \
        --nonTerminal "${SUPERGRAMMAR_SYMBOL}_direct" \
        --to_epoch 50 \
        --subgrammar $SUPERGRAMMAR \
        --train_type new \
        --seed $i
    
    python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
        --model $MODEL \
        --nonTerminal $SUPERGRAMMAR_SYMBOL \
        --to_epoch 50 \
        --subgrammar $SUPERGRAMMAR \
        --train_type continued \
        --seed $i
done

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal "${SUPERGRAMMAR_SYMBOL}_direct" \
    --to_epoch 50 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new \
    --create_table

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model $MODEL \
    --nonTerminal $SUPERGRAMMAR_SYMBOL \
    --to_epoch 50 \
    --subgrammar $SUPERGRAMMAR \
    --train_type continued \
    --create_table
