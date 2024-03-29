#!/bin/sh -l

#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=8G

# Print the hostname of the compute node on which this job is running.
/bin/hostname

python train.py --config conf/linear_regression.yaml

python train.py --config conf/linear_regression.yaml
