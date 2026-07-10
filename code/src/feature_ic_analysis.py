import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from config import config
from utils import engineer_features_158plus39


def compute_ic_analysis(data_path='./data/train.csv'):
    print("=" * 60)
    print("特征IC分析脚本")
    print("=" * 60)
    
    raw_df = pd.read_csv(data_path, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    
    print(f"原始数据: {len(raw_df)} 行")
    
    groups = [group for _, group in raw_df.groupby('股票代码', sort=False)]
    print(f"股票数量: {len(groups)}")
    
    num_processes = min(10, os.cpu_count())
    print(f"使用 {num_processes} 进程进行特征工程...")
    
    from multiprocessing import Pool
    with Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups), total=len(groups), desc='特征工程'))
    
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])
    
    print(f"特征工程后: {len(processed)} 行")
    
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    processed = processed[processed['open_t1'] > 1e-4]
    processed['abs_return'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['market_return'] = processed.groupby('日期')['abs_return'].transform('mean')
    processed['excess_return'] = processed['abs_return'] - processed['market_return']
    
    processed = processed.dropna(subset=['excess_return'])
    
    feature_columns = [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    print(f"\n特征数量: {len(feature_columns)}")
    
    ic_results = []
    
    print("\n计算每日Rank IC...")
    for date, group in tqdm(processed.groupby('日期'), desc='日期循环'):
        if len(group) < 10:
            continue
        
        for feature in feature_columns:
            if feature not in group.columns:
                continue
            
            feature_vals = group[feature].values
            target_vals = group['excess_return'].values
            
            mask = ~np.isnan(feature_vals) & ~np.isnan(target_vals)
            if mask.sum() < 10:
                continue
            
            feature_clean = feature_vals[mask]
            target_clean = target_vals[mask]
            
            try:
                ic, p_value = spearmanr(feature_clean, target_clean)
                ic_results.append({
                    '日期': date,
                    '特征': feature,
                    'IC': ic,
                    'p_value': p_value
                })
            except:
                continue
    
    ic_df = pd.DataFrame(ic_results)
    
    print("\n计算特征统计量...")
    feature_stats = []
    for feature in tqdm(feature_columns, desc='特征统计'):
        feature_ics = ic_df[ic_df['特征'] == feature]['IC'].values
        if len(feature_ics) == 0:
            continue
        
        ic_mean = np.mean(feature_ics)
        ic_std = np.std(feature_ics)
        icir = ic_mean / (ic_std + 1e-12)
        ic_abs_mean = np.mean(np.abs(feature_ics))
        
        positive_ic_ratio = np.mean(feature_ics > 0)
        
        feature_stats.append({
            '特征': feature,
            'IC均值': ic_mean,
            'IC绝对值均值': ic_abs_mean,
            'IC标准差': ic_std,
            'ICIR': icir,
            '正IC比例': positive_ic_ratio,
            '样本数': len(feature_ics)
        })
    
    stats_df = pd.DataFrame(feature_stats)
    stats_df = stats_df.sort_values('IC绝对值均值', ascending=False)
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    stats_df.to_csv(os.path.join(output_dir, 'feature_ic_stats.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("特征IC分析结果（按IC绝对值均值排序）")
    print("=" * 60)
    print(stats_df[['特征', 'IC均值', 'IC绝对值均值', 'IC标准差', 'ICIR', '正IC比例']].head(30).to_string())
    
    print(f"\n完整结果已保存到: {os.path.join(output_dir, 'feature_ic_stats.csv')}")
    
    high_ic_features = stats_df[stats_df['IC绝对值均值'] > 0.02]['特征'].tolist()
    print(f"\n高IC特征(IC绝对值均值>0.02): {len(high_ic_features)} 个")
    print(", ".join(high_ic_features))
    
    return stats_df, high_ic_features


if __name__ == '__main__':
    compute_ic_analysis()
