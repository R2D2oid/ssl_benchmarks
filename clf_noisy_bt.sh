#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --gres=gpu:v100l:1       # Number of GPU(s) per node
#SBATCH --cpus-per-task=8        # CPU cores/threads
#SBATCH --mem=24G                # memory per node
#SBATCH --time=0-2:00            # time (DD-HH:MM)
#SBATCH --output=bt_clf_noisy.out
#SBATCH --array=0-9

source env_ssl/bin/activate

python3 cifar10_noisy.py --user_name='zahrav' --model_name='BarlowTwinsModel' --noise_rate=0.$SLURM_ARRAY_TASK_ID --noise_type='sym'
python3 cifar10_noisy.py --user_name='zahrav' --model_name='BarlowTwinsModel' --noise_rate=0.$SLURM_ARRAY_TASK_ID --noise_type='asym'

