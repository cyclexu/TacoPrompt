#!/bin/bash

#SBATCH --job-name=TaxoComplete
#SBATCH --output=./TaxoComplete/%x-%j.out
#SBATCH --error=./TaxoComplete/%x-%j.err
#SBATCH --time=10-05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G 
#SBATCH --partition=nodes
#SBATCH --gres=gpu:1

# Verify working directory
echo $(pwd)
module load miniconda/3
module load cuda/11.1
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate taxocomplete
python ./src/train.py --config ./config_files/semeval_noun/config_clst20_s47.json
python ./src/train.py --config ./config_files/semeval_noun/config_clst20_s48.json
python ./src/train.py --config ./config_files/semeval_noun/config_clst20_s49.json
python ./src/train.py --config ./config_files/semeval_noun/config_clst20_s50.json
python ./src/train.py --config ./config_files/semeval_noun/config_clst20_s51.json
conda deactivate