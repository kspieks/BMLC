# BMLC: Baseline Machine Learning for Cheminformatics
The goal of this repo is to easily train several baseline ML models from sklearn using various fingerprint representations from RDKit. Optuna is used to vary the hyperparameters and optimize performance of each fingerprint + model combination on the provided validation set.

## Pip installation instructions

```
# create conda env
conda create -n bmlc python=3.11 -y

# activate conda env
conda activate bmlc 

# install rdkit
pip install rdkit

# install other packages
pip install hiplot optuna pandarallel scikit-learn seaborn xgboost

pip install git+https://github.com/bp-kelley/descriptastorus

# install repo in editable mode
pip install -e .
```