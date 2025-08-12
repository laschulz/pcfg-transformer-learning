#!/bin/bash

SUPERGRAMMAR="ABC_grammar"
SUBGRAMMAR_TRAIN="L1_3c_subgrammar"
DATASET_SIZE=300
MODEL="FourLayer"
CONTINUE_FROM=20

cd src
python generate_pcfg.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --start_symbol L0

python generate_pcfg.py --grammar $SUBGRAMMAR_TRAIN --dataset_size $DATASET_SIZE --start_symbol L0 \
    --tokenizer_path "../data/${SUPERGRAMMAR}/${SUPERGRAMMAR}_${DATASET_SIZE}/tokenizer.json"

python weight_space.py --grammar $SUPERGRAMMAR --dataset_size $DATASET_SIZE --model $MODEL \
    --subgrammar $SUBGRAMMAR_TRAIN --num_epochs_direct 40 --num_epochs_pretrain 20


# think about that CKA is focusing on the global structure and might miss local structures / not put as much
# priority on local structures. Apparently RSA is better for that but I mainly see i t in neuroscience papers.
# could also do PWCCA or windowed/conditional CKA, or CKA on nonterminal spans. But gotta think about it more. 