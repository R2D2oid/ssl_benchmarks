#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8              # CPU cores/threads
#SBATCH --mem=24G                      # memory per node
#SBATCH --time=0-3:00                  # time (DD-HH:MM)
#SBATCH --output=ssl.out

source env_ssl/bin/activate

#python3 cifar10_benchmark.py --user_name='srangrej' --model_name='SimCLR'
#python3 cifar10_benchmark.py --user_name='srangrej' --model_name='SwAV'
python3 cifar10_benchmark.py --user_name='zahrav' --model_name='MoCo'
#python3 cifar10_benchmark.py --user_name='zahrav' --model_name='BarlowTwinsModel'
#python3 cifar10_benchmark.py --user_name='iamara' --model_name='SimSiam'
#python3 cifar10_benchmark.py --user_name='iamara' --model_name='BYOL'
