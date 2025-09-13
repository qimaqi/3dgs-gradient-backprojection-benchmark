#!/bin/bash
#SBATCH --job-name=scannetpp_rescale_2
#SBATCH --output=sbatch_log/scannetpp_rescale_2_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,bmicgpu10,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch


cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate 3dgsback
python run_scannetpp.py --rescale 2 --split /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/scannetpp_mini_val.txt

python run_scannetpp.py --rescale 2 --split /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/scannetpp_val.txt