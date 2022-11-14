#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TutorialJob_dino

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=5G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:2
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=15:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=/home/stud100/ex_01_imitation_learning/outputs/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/stud100/ex_01_imitation_learning/outputs/job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=
# your mail address

####
#c) Execute your file.
####

python main.py --train

echo DONE!
