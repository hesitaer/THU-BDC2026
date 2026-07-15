import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import multiprocessing as mp
from utils import engineer_features_corr_filtered
from config import config

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_columns = [
    'instrument', '开盘', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'KMID', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
    'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    'MA5', 'MA10', 'MA20', 'MA30', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
    'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
    'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
    'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
    'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
    'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
    'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30',
    'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
    'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
    'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30',
    'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
    'rsi', 'macd', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_std', 'atr_14',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

def build_label_and_clean(processed, drop_small_open=True):
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['abs_return'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['abs_return'])
    processed['market_return'] = processed.groupby('日期')['abs_return'].transform('mean')
    processed['excess_return'] = processed['abs_return'] - processed['market_return']
    
    return processed

def preprocess_data(df, stockid2idx=None, drop_small_open=True):
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError("输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        from tqdm import tqdm
        processed_list = list(tqdm(pool.imap(engineer_features_corr_filtered, groups), total=len(groups), desc="特征工程"))

    processed = pd.concat(processed_list).reset_index(drop=True)

    if stockid2idx is not None:
        processed['instrument'] = processed['股票代码'].map(stockid2idx)
        processed = processed.dropna(subset=['instrument']).copy()
        processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed

def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    train_df = df[df['日期'] < val_start].copy()
    val_df = df.copy()

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

def calculate_ranking_metrics(y_pred, y_true, group, k=5):
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    final_score_list = []

    idx = 0
    for g in group:
        if g < k:
            idx += g
            continue

        pred = y_pred[idx:idx+g]
        true = y_true[idx:idx+g]

        pred_indices = np.argsort(pred)[::-1][:k]
        pred_top_returns = true[pred_indices]
        pred_return_sum = pred_top_returns.sum()

        true_indices = np.argsort(true)[::-1][:k]
        true_top_returns = true[true_indices]
        max_return_sum = true_top_returns.sum()

        random_return_sum = k * true.mean()

        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        final_score_list.append(final_score)

        idx += g

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'top5_return': np.mean(pred_return_sum_list) / k if pred_return_sum_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
    }

    return metrics

def train_lgb_lambdarank():
    set_seed(config.get('seed', 42))
    output_dir = os.path.join('./model', 'lgb_lambdarank_baseline')
    os.makedirs(output_dir, exist_ok=True)

    data_path = config['data_path']
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}

    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    train_data = preprocess_data(train_df, stockid2idx=stockid2idx, drop_small_open=True)
    val_data = preprocess_data(val_df, stockid2idx=stockid2idx, drop_small_open=True)

    features = [col for col in feature_columns if col != 'instrument']

    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    val_start_str = val_start.strftime('%Y-%m-%d')
    val_data = val_data[val_data['日期'] >= val_start_str].copy()

    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"特征数量: {len(features)}")

    train_group = train_data.groupby('日期').size().values
    val_group = val_data.groupby('日期').size().values

    assert np.sum(train_group) == len(train_data), "训练集group求和与样本数不一致"
    assert np.sum(val_group) == len(val_data), "验证集group求和与样本数不一致"

    train_group_size = train_data.groupby('日期')['excess_return'].transform('size')
    train_rank = train_data.groupby('日期')['excess_return'].rank(method='first', ascending=True).astype(np.int32)
    train_data['label'] = (train_group_size - train_rank + 1).astype(np.int32)
    
    val_group_size = val_data.groupby('日期')['excess_return'].transform('size')
    val_rank = val_data.groupby('日期')['excess_return'].rank(method='first', ascending=True).astype(np.int32)
    val_data['label'] = (val_group_size - val_rank + 1).astype(np.int32)

    X_train = train_data[features].values
    y_train = train_data['label'].values
    y_train_return = train_data['abs_return'].values

    X_val = val_data[features].values
    y_val = val_data['label'].values
    y_val_return = val_data['abs_return'].values

    max_label = max(y_train.max(), y_val.max()) if len(y_train) > 0 and len(y_val) > 0 else 100
    label_gain = list(range(1, int(max_label) + 2))

    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_group)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=val_group, reference=lgb_train)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 1,
        'seed': 42,
        'num_threads': mp.cpu_count(),
        'ndcg_eval_at': [5, 10, 20],
        'max_depth': -1,
        'min_child_weight': 1,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'min_data_in_leaf': 20,
        'label_gain': label_gain,
    }

    print("\n开始训练LightGBM LambdaRank...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        valid_names=['valid'],
        callbacks=callbacks,
    )

    print("\n训练完成！最佳迭代次数:", model.best_iteration)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_metrics = calculate_ranking_metrics(y_pred_train, y_train_return, train_group, k=5)
    val_metrics = calculate_ranking_metrics(y_pred_val, y_val_return, val_group, k=5)

    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n验证集指标:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    model.save_model(os.path.join(output_dir, 'lgb_lambdarank_model.txt'))
    joblib.dump(scaler, os.path.join(output_dir, 'lgb_scaler.pkl'))

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best iteration: {model.best_iteration}\n")
        f.write(f"Best final_score: {val_metrics['final_score']:.6f}\n")

    print(f"\n模型已保存到: {output_dir}")

    return val_metrics['final_score']

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = train_lgb_lambdarank()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")