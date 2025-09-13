#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/convformer0_2layer_acdc_inception_conv_elu_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate 3dgsback_gsplat14
conda activate 3dgsback