#!/bin/bash
##
## ntasks = numbers of cores
#SBATCH --ntasks=40
#SBATCH --gres=gpu:1

#
## time = walltime
#SBATCH --time=11-12:00:00
#
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebij1910@gmail.com
#
## Job name. It is shown in 'squeue' command output.
## It is recommended, but not necessary, that the job name be the same
## that the input file name
#SBATCH --job-name=deepscreen_trypanosomatics_train
echo -e "\n##################### Job started in $(date +'%d-%m-%Y as %T') #####################\n"
## The input and output file names are based in job name
## It is not mandatory but helps keeping things tidy
INP=$SLURM_JOB_NAME".inp"
## Job information printed in output file
echo -e "\n Active job of $USER: \n"
squeue -a -u $USER
echo -e "\n Job execution node: $(hostname -s) \n"
echo -e "\n Number of tasks for this job: $SLURM_NTASKS \n"

#########################################
##------- Start of job ----- #
#########################################
## Configure the software environment

module load anaconda3

## Informations about the execution environment printed in output file
echo -e "\n Job submission directory: $SLURM_SUBMIT_DIR \n"
echo -e "\n Job scratch directory: $SCRATCH \n"
echo -e "\n Input file: $INP \n"

## Program run.
cd $SLURM_SUBMIT_DIR
cd ./src/ml
python3 train_search.py

echo -e "\n################### Job finished in $(date +'%d-%m-%Y as %T') ###################"

#########################################
##------- End of job ----- #
#########################################