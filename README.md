### Environment Setup
```
module load python/3.6
virtualenv --system-site-packages -p python3 env_ssl
source env_ssl/bin/activate
pip install -r requirements.txt
```

### Train SSL Model on CIFAR10
```
sbatch ssl_cifar10_sbatch.sh 
```

### Train SSL Model on TinyImagenet
```
sbatch ssl_tiny_sbatch.sh 
```

### Train Classifier using various noise-rates
```
sbatch clf_noisy_bt.sh
```
