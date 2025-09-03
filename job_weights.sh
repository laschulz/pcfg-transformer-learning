#!/bin/bash -l

SUPERGRAMMAR="PythonPCFG"
SUPERGRAMMAR_SYMBOL="STMTS"
SUBGRAMMAR_SYMBOL="compound_stmt"
DATASET_SIZE=50000 #50k
MODEL="OneLayer"

#cd /om2/user/laschulz/pcfg-transformer-learning/src
cd src

# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUPERGRAMMAR_SYMBOL --max_len 250

# python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol $SUBGRAMMAR_SYMBOL \
#     --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}_${SUPERGRAMMAR_SYMBOL}/tokenizer.json" --max_len 250

python weight_space.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL \
    --grammar_startsymbol $SUPERGRAMMAR_SYMBOL --subgrammar_startsymbol $SUBGRAMMAR_SYMBOL --num_epochs_direct 45 --num_epochs_pretrain 5 --start_seed 30

# python generate_pcfg.py --grammar PythonPCFG --dataset_size 50000 --start_symbol STMTS --max_len 250

# python generate_pcfg.py --grammar PythonPCFG --dataset_size 50000 --start_symbol compound_stmt \
#   --tokenizer_path "../data/PythonPCFG/PythonPCFG_50000_STMTS/tokenizer.json" --max_len 250

# python weight_space.py --grammar PythonPCFG --dataset_size 50000 --model TwoLayer \
#   --grammar_startsymbol STMTS --subgrammar_startsymbol compound_stmt --num_epochs_direct 30 --num_epochs_pretrain 3 --start_seed 19
