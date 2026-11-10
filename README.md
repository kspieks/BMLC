# BMLC: Baseline Machine Learning for Cheminformatics

Trains baseline ML models from sklearn on various fingerprints and descriptors from RDKit:

- **Featurizers**: Morgan, atom pair, donor pair, topological torsion, RDKit path, Avalon, MACCS, MQN, and RDKit 2D descriptors (raw or normalized, via descriptastorus). Featurizers can be concatenated, and reactions are featurized as `[reactant FP, product FP - reactant FP]`.
- **Models**: Lasso, Ridge, LinearSVR, SVR, PLS, MLP, random forest, and XGBoost.
- **Hyperparameter search**: Optuna (TPE sampler) minimizes the mean validation RMSE over the provided splits, then the best model is refit on each split.
- **Prediction**: an ensemble of the per-split models, reported as the mean and standard deviation.

## Main commands

```bash
# once, after cloning
brew install uv libomp
uv sync

# train on a csv with precomputed splits
cd examples/ESOL
uv run bash train.sh

# predict with the trained models
uv run bash predict.sh

# open the notebooks
uv run jupyter lab
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, on macOS, the OpenMP runtime used by
XGBoost: `brew install uv libomp`. Without `libomp`, `import xgboost` fails with
`Library not loaded: @rpath/libomp.dylib`. Python 3.12; minimum versions are declared in
`pyproject.toml` and the exact versions used are locked in `uv.lock`.

```bash
uv sync    # creates .venv and installs the package in editable mode
```

`uv sync` also installs the packages needed for the notebooks (JupyterLab, ipykernel,
astartes, hiplot). To skip them, use `uv sync --no-default-groups`.

To run `python` without the `uv run` prefix, activate the environment once per terminal:

```bash
source .venv/bin/activate
```

## Train

`train.py` reads a csv with a SMILES column and a target column, plus a pickle of splits:
a list of `[train_indices, val_indices, test_indices]`, one entry per split.
Featurizer settings come from a yaml file; see `examples/ESOL/featurizer_settings.yaml`.

```bash
uv run python train.py \
    --save_dir results \
    --data_path data.csv \
    --target logSolubility \
    --split_path splits.pkl \
    --featurizer_yaml_path featurizer_settings.yaml \
    --models RF XGB \
    --n_trials 30 \
    --n_cpus_optuna 4
```

| Option | Default | Notes |
|---|---|---|
| `--smiles_column` | `smiles` | |
| `--rxn_mode` | off | SMILES are reactions, `reactants>>products` |
| `--models` | | one or more of `Lasso`, `LinearSVR`, `SVR`, `MLP`, `PLS`, `Ridge`, `RF`, `XGB` |
| `--n_trials` | 32 | Optuna trials per featurizer and model |
| `--n_cpus_optuna` | 4 | trials run in parallel; results are only reproducible run to run with 1 |
| `--n_cpus_featurize` | 2 | |
| `--random_state` | 42 | |

For each featurizer and model, `save_dir` gets `{model}_{name}_best_models.pkl`,
`_y_scalers.pkl`, `_summary.csv`, `_predictions.csv`, and `_optuna_results.csv`, plus
`{name}_settings.yaml` for prediction and `naive_baseline_summary.csv`.

## Predict

```bash
uv run python predict.py \
    --save_dir results \
    --data_path new_data.csv \
    --model_path results/RF_MACCS_best_models.pkl \
    --y_scaler_path results/RF_MACCS_y_scalers.pkl \
    --featurizer_yaml_path results/MACCS_settings.yaml \
    --pred_name predictions.csv
```

The output csv has `mean_pred`, `ensemble_std`, and one column per split model.

## Examples

| Folder | Dataset | Scripts |
|---|---|---|
| `examples/ESOL` | aqueous solubility, 1,128 molecules, scaffold and random splits | `train.sh`, `predict.sh` |
| `examples/RDB7` | reaction barrier `dE0`, 23,854 reactions (forward and reverse), random splits | `train.sh`, `predict.sh` |

Run each script from its own folder, e.g. `cd examples/RDB7 && uv run bash train.sh`.

### RDKit 2026.03.1 changed `NumHAcceptors`

RDKit 2026.03.1 changed the H-bond acceptor SMARTS used by `NumHAcceptors`
(`rdMolDescriptors.CalcNumHBA`, version 2.0.1 to 2.0.2) so that aromatic nitrogens with
three connections, such as the N in N-methylpyrrole or N-methylpyridone, are no longer
counted. Caffeine goes from 6 acceptors to 3, which matches PubChem. The change is
[rdkit issue #8997](https://github.com/rdkit/rdkit/issues/8997), fixed in
[rdkit PR #9060](https://github.com/rdkit/rdkit/pull/9060) and listed under Bug Fixes in the
[Release_2026_03_1 notes](https://github.com/rdkit/rdkit/releases/tag/Release_2026_03_1).

As a result, the `rdkit_2d` and `rdkit_2d_normalized` features differ from those made with
older RDKit versions for any molecule containing such a nitrogen (37 of the 1,128 ESOL
molecules). descriptastorus still normalizes this column with a distribution fitted to the
old definition.

## Notebooks

```bash
uv run jupyter lab
```

In VS Code, select `.venv/bin/python` as the notebook kernel instead.

| Notebook | Contents |
|---|---|
| `examples/ESOL/splits/create_splits_astartes.ipynb` | Creates the ESOL random and scaffold splits; rerunning overwrites the two split pickles |
| `examples/ESOL/UQ_RDKFingerprint_Tanimoto.ipynb` | Whether Tanimoto similarity to the training set predicts error on split 0 |
| `ipython/compare_fingerprint_generators.ipynb` | Checks that different RDKit APIs give the same Morgan and atom pair fingerprints |
| `ipython/parallel_plot.ipynb` | HiPlot parallel-coordinates plot of Optuna results |

## Updating package versions

```bash
uv lock --upgrade   # newest versions allowed by pyproject.toml
uv sync
```

Rerun the examples and notebooks to check the outputs, then commit the updated `uv.lock`.

## Project structure

```
├── baseline_models/
│   ├── main.py                 # BaselineML: featurize, Optuna search, refit, save
│   ├── training.py             # train_model, Optuna objective, naive baseline, metrics
│   ├── models.py               # model constructors and their search spaces
│   ├── features/               # featurizers and create_features
│   ├── plotting_utils/         # parity, residual, and classification plots
│   └── utils/                  # CLI parsing, logging, yaml
├── train.py                    # CLI: train
├── predict.py                  # CLI: predict
├── examples/                   # ESOL and RDB7 data, splits, scripts, and results
├── ipython/                    # notebooks
├── pyproject.toml
└── uv.lock
```
