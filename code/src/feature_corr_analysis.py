import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import config
from utils import engineer_features_158plus39


def compute_corr_analysis(data_path='./data/train.csv'):
    print("=" * 60)
    print("特征相关性分析脚本")
    print("=" * 60)
    
    raw_df = pd.read_csv(data_path, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    
    print(f"原始数据: {len(raw_df)} 行")
    
    groups = [group for _, group in raw_df.groupby('股票代码', sort=False)]
    print(f"股票数量: {len(groups)}")
    
    from multiprocessing import Pool
    num_processes = min(10, os.cpu_count())
    with Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups), total=len(groups), desc='特征工程'))
    
    processed = pd.concat(processed_list).reset_index(drop=True)
    
    feature_columns = [col for col in processed.columns if col not in ['股票代码', '日期', 'instrument']]
    print(f"特征数量: {len(feature_columns)}")
    
    print("\n计算特征相关性矩阵...")
    corr_matrix = processed[feature_columns].corr()
    
    high_corr_pairs = []
    threshold = 0.9
    
    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                high_corr_pairs.append({
                    '特征1': feature_columns[i],
                    '特征2': feature_columns[j],
                    '相关系数': corr
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('相关系数', ascending=False)
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    high_corr_df.to_csv(os.path.join(output_dir, 'feature_correlations.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n高度相关特征对(相关系数>0.9): {len(high_corr_df)} 对")
    print(high_corr_df.head(30).to_string())
    
    print(f"\n完整结果已保存到: {os.path.join(output_dir, 'feature_correlations.csv')}")
    
    features_to_remove = set()
    for _, row in high_corr_df.iterrows():
        if row['相关系数'] > 0.95:
            features_to_remove.add(row['特征2'])
    
    print(f"\n建议移除的冗余特征(相关系数>0.95): {len(features_to_remove)} 个")
    print(", ".join(sorted(features_to_remove)))
    
    return high_corr_df, features_to_remove


if __name__ == '__main__':
    compute_corr_analysis()
