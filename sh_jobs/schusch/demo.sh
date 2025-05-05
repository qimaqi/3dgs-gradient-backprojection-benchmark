#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --output=sbatch_log/demo_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate 3dgsback

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark


python backproject.py