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
mkdir -p /om2/user/laschulz/pcfg_transformer-learning/logs

cd /om2/user/laschulz/pcfg_transformer-learning || exit 1

PCFGS=("CenterEmbedding" "ArithmeticLogic")
NUM_PCFG=${#PCFGS[@]}
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_PCFG))

PCFG = "${PCFGS[$SEED_INDEX]}"

echo "Running with PCFG:

python src/main.py \
    --mode multiple \
    --teacher_type baselineCNN \
    --student_type multiChannelCNN \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp2
