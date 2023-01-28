# Installation
Install the packages listed in `requirements.txt`.  Source code relevant to this project is found in the `src` folder

via conda
* make a new environment with name "FEGP": 
```
conda create -n FEGP
```
* activate new environment: 
```
conda activate FEGP
```
* install packages into environment: 
```
conda install --file requirements.txt
```

# Basics
Gaussian process functionality provided by `gpytorch`, bayesian optimization framework provided by `botorch`.  Acqusition functions were written to fit with the `botorch` API.  

# Preprocessing
To run the data preprocessing script, simply change `dir = '/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/KHM010_'` in `src\preprocess.py` to reflect your local repo and run. The processed data will be saved to a subdirectory processed in the data folder. 

# Training
To run the Gaussian Process training script, navigate to `src\main.py.` You will need the `wandb` module, which can be installed via pip. Refer to https://docs.wandb.ai/quickstart for info on start-up and https://docs.wandb.ai/guides/sweeps/quickstart for how to conduct a sweep.   
