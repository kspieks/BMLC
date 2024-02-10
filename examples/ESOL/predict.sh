BASELINE_MODELS='../../../BMLC'

save_dir=`pwd`
echo $save_dir
log_name='pred'

data_path='data/delaney.csv'
model_path='RF_rdkit_2d_normalized_best_models.pkl'
y_scaler_path='RF_rdkit_2d_normalized_y_scalers.pkl'

featurizer_yaml_path=rdkit_2d_normalized_settings.yaml
n_cpus_featurize=4

pred_name='RF_rdkit_2d_normalized_test_predictions.csv'


python $BASELINE_MODELS/predict.py \
--log_name $log_name \
--save_dir $save_dir \
--data_path $data_path \
--model_path $model_path \
--y_scaler_path $y_scaler_path \
--n_cpus_featurize $n_cpus_featurize \
--featurizer_yaml_path $featurizer_yaml_path \
--pred_name $pred_name
