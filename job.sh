#!/bin/bash                      
#SBATCH -t 30:00:00                 
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp2
#SBATCH --array=0-29               
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_%A_%a.log

hostname                             # Print node info

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate pcfg_transformer

# Create logs directory if needed
mkdir -p /om2/user/laschulz/pcfg-transformer-learning/logs

cd /om2/user/laschulz/pcfg-transformer-learning

PCFGS=("CenterEmbedding" "ArithmeticLogic")
NUM_PCFG=${#PCFGS[@]}
INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_PCFG ))

PCFG="${PCFGS[$INDEX]}"

# concatenate pcfgs with + 
IFS='+' 
LIST_PCFG="${PCFGS[*]}"   # now LIST_PCFG="CenterEmbedding+ArithmeticLogic"
unset IFS

python src/main.py \
    --pcfg $PCFG \
    --dataset "$PCFG{_5000}" \
    --model FourLayer


python src/analysis.py \
    --pcfg $LIST_PCFG \
    --dataset_size 5000 \
    --model FourLayer 
