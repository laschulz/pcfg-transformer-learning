SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR1="PythonPCFG_symbol"
SUPERGRAMMAR_SYMBOL="STMTS"
SUBGRAMMAR_SYMBOL="compound_stmt"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer"
NUM_EPOCHS=40 #40
SEED=2

cd src
#baseline
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# # LM with symbol
# python generate_pcfg.py --grammar $SUPERGRAMMAR1 --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# # LM with just symbol
# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUBGRAMMAR_SYMBOL --max_len 250

# # # ------- Training -------

# python train.py --grammar $SUPERGRAMMAR --dataset_name "${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}" \
#         --model $MODEL --num_epochs $NUM_EPOCHS --seed $SEED

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        # --dataset_name "${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}" \
        # --model $MODEL \
        # --nonTerminal $SUBGRAMMAR_SYMBOL \
        # --to_epoch 50 \
        # --subgrammar $SUPERGRAMMAR \
        # --train_type new \
        # --seed $SEED

# python train.py --grammar $SUPERGRAMMAR1 \
#         --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#         --model $MODEL \
#         --num_epochs $NUM_EPOCHS \
#         --seed $SEED

# python analysis_hierarchy.py --grammar $SUPERGRAMMAR1 \
#         --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#         --model $MODEL \
#         --nonTerminal $SUPERGRAMMAR_SYMBOL \
#         --to_epoch 50 \
#         --subgrammar $SUPERGRAMMAR1 \
#         --train_type new \
#         --seed $SEED

# BASELINE
# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model "TwoLayer_LARGER" \
#     --num_epochs $NUM_EPOCHS \
#     --seed $SEED

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
    --model "TwoLayer_LARGER" \
    --nonTerminal "${SUPERGRAMMAR_SYMBOL}_direct" \
    --to_epoch 50 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new \
    --seed $SEED \
    --plot_only
    
