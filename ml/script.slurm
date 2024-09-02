#!/bin/bash
##
## ntasks = numbers of cores
#SBATCH --ntasks-per-node=20
#SBATCH --nodelist=c008

#
## time = walltime
#SBATCH --time=4-12:00:00
#
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebij1910@gmail.com
#
## Job name. It is shown in 'squeue' command output.
## It is recommended, but not necessary, that the job name be the same
## that the input file name
#SBATCH --job-name=Trypanodeepscreen
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
source /home/sjinich/.conda/envs/trypanodeepscreen_che_env/bin/activate
which python

## Informations about the execution environment printed in output file
echo -e "\n Job submission directory: $SLURM_SUBMIT_DIR \n"
echo -e "\n Job scratch directory: $SCRATCH \n"
echo -e "\n Input file: $INP \n"

## Program run.
cd $SLURM_SUBMIT_DIR
python3 main.py \
--data_train_val_test ../data/processed/CHEMBL4567.csv \
--experiment_result_path ../experiments/ \
--target_name_experiment chembl4567_auroc \
--data_split_mode non_random_split


echo -e "\n################### Job finished in $(date +'%d-%m-%Y as %T') ###################"

#########################################
##------- End of job ----- #
#########################################
