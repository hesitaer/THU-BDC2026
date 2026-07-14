sequence_length = 60
feature_num = 'corr_filtered'
config = {
    'sequence_length': sequence_length,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 4,
    'dim_feedforward': 512,
    'batch_size': 4,
    'num_epochs': 80,
    'learning_rate': 3e-5,
    'dropout': 0.15,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'reg_weight': 0.1,

    'output_dir': f'./model/{sequence_length}_{feature_num}_enhanced',
    'data_path': './data',
    
    'rolling_validation': False,
}