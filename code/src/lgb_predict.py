import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import multiprocessing as mp
from utils import engineer_features_corr_filtered
from config import config

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

def preprocess_predict_data(df, stockid2idx):
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(10, mp.cpu_count())
    from tqdm import tqdm
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_corr_filtered, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    return processed

def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_dir = os.path.join('./model', 'lgb_lambdarank_baseline')
    model_path = os.path.join(model_dir, 'lgb_lambdarank_model.txt')
    scaler_path = os.path.join(model_dir, 'lgb_scaler.pkl')
    output_path = os.path.join('./output/', 'result.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed = preprocess_predict_data(raw_df, stockid2idx)

    features = [col for col in feature_columns if col != 'instrument']
    latest_data = processed[processed['日期'] == latest_date].copy()

    if len(latest_data) == 0:
        raise ValueError(f'最新日期 {latest_date} 没有数据')

    scaler = joblib.load(scaler_path)
    latest_data[features] = latest_data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest_data[features] = scaler.transform(latest_data[features])

    model = lgb.Booster(model_file=model_path)
    scores = model.predict(latest_data[features].values)

    latest_data['score'] = scores
    latest_data = latest_data.sort_values('score', ascending=False).reset_index(drop=True)

    top5 = latest_data.head(5)
    top5_stock_ids = top5['股票代码'].tolist()
    top5_scores = top5['score'].values

    top5_scores = top5_scores - np.min(top5_scores) + 1e-12
    weights = np.exp(top5_scores) / np.sum(np.exp(top5_scores))

    min_weight = 0.02
    max_weight = 0.3
    weights = np.maximum(weights, min_weight)
    weights = np.minimum(weights, max_weight)
    weights = weights / np.sum(weights)

    os.makedirs('./output/', exist_ok=True)
    output_df = pd.DataFrame({
        'stock_id': top5_stock_ids,
        'weight': weights,
    })
    output_df.to_csv(output_path, index=False)

    print(f'预测日期: {latest_date.date()}')
    print(f'参与排序股票数: {len(latest_data)}')
    print(f'Top5股票及权重:')
    for sid, w in zip(top5_stock_ids, weights):
        print(f'  {sid}: {w:.4f}')
    print(f'结果已写入: {output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
