#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=03-08
#SBATCH --job-name=run_Train_MOTIF_CNN_pipeline
#SBATCH --output=run_Train_MOTIF_CNN_pipeline.o%j
#SBATCH --error=run_Train_MOTIF_CNN_pipeline.e%j



### Functions

# pre-process
function processing_module {
    local job_id=($(sbatch --export=configfile=$configFile --mem=${process_classification_mem} --time=${process_classification_time} --mail-user=${user}  ${process_classification_Path}))
    echo ${job_id[3]}
}

function build_network_module {
    local job_id=($(sbatch --export=configfile=$configFile --depend=afterok:$1  --mem=${build_network_mem} --time=${build_network_time} --mail-user=${user}} ${build_network_Path}))
    echo ${job_id[3]}
}
function build_network_module {
    local job_id=($(sbatch --export=configfile=$configFile   --mem=${build_network_mem} --time=${build_network_time} --mail-user=${user}} ${build_network_Path}))
    echo ${job_id[3]}
}



function eval_network_module {
    local job_id=($(sbatch --export=configfile=$configFile --depend=afterok:$1  --mem=${eval_network_mem} --time=${eval_network_time} --mail-user=${user}} ${eval_network_Path}))
    echo ${job_id[3]}
}
function eval_network_module {
    local job_id=($(sbatch --export=configfile=$configFile  --mem=${eval_network_mem} --time=${eval_network_time} --mail-user=${user}} ${eval_network_Path}))
    echo ${job_id[3]}
}



function train_eval_network_module {
    local job_id=($(sbatch --export=configfile=$configFile  --mem=${eval_network_mem} --time=${eval_network_time} --mail-user=${user} ${train_eval_network_Path}))
    echo ${job_id[3]}
}




source configurationFile.txt

mkdir -p ${data_path}
mkdir -p $OutputLogs
mkdir -p $ErrorLogs

# defining dependencies:
#echo Inititaing preprocessing module at `date`
#preprocessing_job=$(processing_module)
#echo Finished preprocessing data at `date`

#echo Initiating model building  at `date`
#build_network_job=$(build_network_module  ${preprocessing_job})
#echo finished building and training network at `date`

echo Initiating model evaluation  at `date`
#eval_network_job=$(eval_network_module  ${build_network_job})
train_eval_network_job=$(train_eval_network_module )
echo finished evaluatin network at `date`

mv *.o* $OutputLogs
mv *.e* $ErrorLogs
