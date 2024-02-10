
BASELINE_MODELS="../../../BMLC"

save_dir=training_results
echo $save_dir
log_name=train

target="dE0"
data_path="data_splits/fwd_rev/RDB7_fwd_rev_rxns.csv"
split_path="data_splits/fwd_rev/rxn_random_split_seed_0_1_2_3_4.pkl"
smiles_column="rxn_smiles"

featurizer_yaml_path=featurizer_settings.yaml
models="LinearSVR XGB"

n_cpus_optuna=5
n_trials=25

python $BASELINE_MODELS/train.py \
--log_name $log_name \
--save_dir $save_dir \
--data_path $data_path \
--smiles_column $smiles_column \
--rxn_mode \
--target $target \
--split_path $split_path \
--featurizer_yaml_path $featurizer_yaml_path \
--models $models \
--n_cpus_optuna $n_cpus_optuna \
--n_trials $n_trials
