#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1                 # Request 1 GPU (remove if you want CPU-only)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=Pythongram
#SBATCH --array=1-3                   # Run 3 parallel jobs (array indices 1,2,3)
#SBATCH --output=/om2/user/laschulz/pcfg-transformer-learning/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/pcfg-transformer-learning/logs/error_%A_%a.log

hostname

# Activate conda
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate pcfg

# Create logs directory if needed
mkdir -p /om2/user/laschulz/pcfg-transformer-learning/logs

# Parameters
SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
SUBGRAMMAR_SYMBOL="compound_stmt"
DATASET_SIZE=50000
MODEL="TwoLayer"
CONTINUE_FROM=3
NUM_EPOCHS=40

cd /om2/user/laschulz/pcfg-transformer-learning/src


# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUBGRAMMAR_SYMBOL \
#     --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

# # ------- Training -------
for SEED in 3; do
    python train.py --grammar $SUPERGRAMMAR --dataset_name \"${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}\" \
        --model $MODEL --num_epochs $CONTINUE_FROM --seed $SEED

    python train.py --grammar $SUPERGRAMMAR \
        --dataset_name \"${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}\" \
        --model $MODEL \
        --continue_training \
        --checkpoint_path \"${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUBGRAMMAR_SYMBOL}/${MODEL}/new/seed_${SEED}/epoch_${CONTINUE_FROM}_0.pt\" \
        --num_epochs $NUM_EPOCHS \
        --continue_from $CONTINUE_FROM \
        --seed $SEED

    python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
        --dataset_name \"${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}\" \
        --model $MODEL \
        --nonTerminal $SUPERGRAMMAR_SYMBOL \
        --to_epoch 50 \
        --subgrammar $SUPERGRAMMAR \
        --train_type continued \
        --seed $SEED
done

# python train.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --num_epochs $((CONTINUE_FROM + NUM_EPOCHS)) \
#     --seed 3


# python analysis_hierarchy.py --grammar $SUPERGRAMMAR \
#     --dataset_name "${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}" \
#     --model $MODEL \
#     --nonTerminal "${SUPERGRAMMAR_SYMBOL}_direct" \
#     --to_epoch 50 \
#     --subgrammar $SUPERGRAMMAR \
#     --train_type new \
#     --seed 3