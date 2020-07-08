#!/bin/bash
#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --job-name=prcess_classification
#SBATCH --output=process_classifcation.o%j
#SBATCH --error=process_classification.e%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --time=00-04
#SBATCH --mem=100GB

module load python/gpu/3.6.5
module load bedtools/2.27.1 

source configurationFile.txt


python process_classification_dataset.py ${window} ${experiment} ${data_path} ${pos_filename} ${neg_filename} ${genome_filename}
