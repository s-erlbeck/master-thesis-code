#!/bin/bash

#SBATCH --partition=becks,lopri
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=90:00
#SBATCH --job-name=SOME-NAME
#SBATCH --output=../logs/job-%x.out

python ../training.py later_tests -lc SOME_CONFIG
