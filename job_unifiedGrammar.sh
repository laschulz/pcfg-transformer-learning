SUPERGRAMMAR="unifiedGrammar"
SUBGRAMMAR="uni_simple_subgrammar"
DATASET_SIZE=300
MODEL="TwoLayer"
NUM_EPOCHS=70

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L0 

python train.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL


python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L0 \
    --to_epoch 100 \
    --subgrammar $SUPERGRAMMAR \
    --train_type new

# all of this doesn't make sense, need to adjust code

python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
    --dataset_size $DATASET_SIZE \
    --model $MODEL \
    --nonTerminal L0 \
    --to_epoch 100 \
    --subgrammar $SUBGRAMMAR \
    --train_type new