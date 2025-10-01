#!/bin/bash -l

SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
SUBGRAMMAR_SYMBOL="compound_stmt"
DATASET_SIZE=50000 #50k
MODEL="FourLayer"

cd ../src

python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUBGRAMMAR_SYMBOL \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

python weight_space.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL \
    --grammar_startsymbol $SUPERGRAMMAR_SYMBOL --subgrammar_startsymbol $SUBGRAMMAR_SYMBOL --num_epochs_direct 40 --num_epochs_pretrain 10 --start_seed 20
