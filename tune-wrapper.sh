#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1

scratch_dir=/data1/slurm/$SLURM_JOB_ID
hpvm_image=$HOME/hpvm.sb
script=$HOME/hpvm-tuning-android/tune_x.py

cd $HOME/hpvm-tuning-android

# singularity exec --nv --writable-tmpfs $hpvm_image python -u tune_x.py resnet18_cifar10 10000 3.0 5.0 -D tuning.resnet18_cifar10 -M tuner_results -B 500
singularity exec --nv --writable-tmpfs $hpvm_image python -u tune_x.py -D $(mktemp -ud) "$@"
