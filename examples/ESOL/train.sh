
BASELINE_MODELS="../../../BMLC"

save_dir=`pwd`
echo $save_dir
log_name="train"

data_path="data/delaney.csv"
target="logSolubility"
split_path="splits/delaney_splits_scaffold.pkl"

featurizer_yaml_path=featurizer_settings.yaml
models="LinearSVR RF"

n_cpus_optuna=5
n_trials=30

python $BASELINE_MODELS/train.py \
--log_name $log_name \
--save_dir $save_dir \
--data_path $data_path \
--target $target \
--split_path $split_path \
--featurizer_yaml_path $featurizer_yaml_path \
--models $models \
--n_cpus_optuna $n_cpus_optuna \
--n_trials $n_trials
