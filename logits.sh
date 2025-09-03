SUPERGRAMMAR="OverlappingSubgrammar"
SUPERGRAMMAR_SYMBOL="L0"
DATASET_SIZE=50000 #50k
MODEL="TwoLayer"


cd src

# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs 40 \
#     --seed 42

python activation_space.py --model $MODEL --checkpoint "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/${MODEL}/new/seed_42/epoch_40_0.pt" \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}"