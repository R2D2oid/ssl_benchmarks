#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8              # CPU cores/threads
#SBATCH --mem=100G                      # memory per node
#SBATCH --time=0-10:00                  # time (DD-HH:MM)
#SBATCH --output=ssl_MoCo_tiny.out

source env_ssl/bin/activate

#python3 tinyimage_benchmarks.py --user_name='srangrej' --model_name='SimCLR'
#python3 tinyimage_benchmarks.py --user_name='srangrej' --model_name='SwAV'
#python3 tinyimage_benchmarks.py --user_name='zahrav' --model_name='BarlowTwinsModel' --data '~/projects/def-jjclark/shared_data/tinyimagenet/data/tiny-imagenet-200'
python3 tinyimage_benchmarks.py --user_name='zahrav' --model_name='MoCo' --data '~/projects/def-jjclark/shared_data/tinyimagenet/data/tiny-imagenet-200'
#python3 tinyimage_benchmarks.py --user_name='iamara' --model_name='SimSiam'
#python3 tinyimage_benchmarks.py --user_name='iamara' --model_name='BYOL'
