BASELINE_MODELS='../../../BMLC'

save_dir=`pwd`
echo $save_dir
log_name='pred'

data_path="data_splits/fwd_rev/RDB7_fwd_rev_rxns.csv"
smiles_column="rxn_smiles"

model_path='training_results/XGB_MACCS_best_models.pkl'
y_scaler_path='training_results/XGB_MACCS_y_scalers.pkl'

featurizer_yaml_path=training_results/MACCS_settings.yaml
n_cpus_featurize=2

pred_name='XGB_MACCS_test_predictions.csv'


python $BASELINE_MODELS/predict.py \
--log_name $log_name \
--save_dir $save_dir \
--data_path $data_path \
--smiles_column $smiles_column \
--rxn_mode \
--model_path $model_path \
--y_scaler_path $y_scaler_path \
--n_cpus_featurize $n_cpus_featurize \
--featurizer_yaml_path $featurizer_yaml_path \
--pred_name $pred_name
