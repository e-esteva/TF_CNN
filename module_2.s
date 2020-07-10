#!/bin/bash
#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=analyize_classification_0
#SBATCH --output=analyze_classifcation.o%j
#SBATCH --error=analyze_classification.e%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --time=00-04
#SBATCH --mem=100GB

module load meme/5.0.2 
module load anaconda3/gpu/5.2.0
conda activate tensorflow2.2
export PYTHONPATH=/gpfs/share/apps/anaconda3/gpu/5.2.0/envs/tensorflow2.2/lib/python3.8/site-packages:$PYTHONPATH

source configurationFile.txt


file_name=${experiment}_${window}.h5

echo ${file_name}
echo 'beginning analysis stage at' `date`

python module_2.py ${data_path} ${file_name}

echo 'analysis module finished at' `date`
