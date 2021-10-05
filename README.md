### Environment Setup
```
module load python/3.6
virtualenv --system-site-packages -p python3 env_ssl
source env_ssl/bin/activate
pip install -r requirements.txt
```

### Train SSL Model
```
sbatch sbatch_script.sh 
```
