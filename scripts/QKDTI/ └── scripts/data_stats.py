# QKDTI/scripts/data_stats.py
"""
Script: data_stats.py
Description: Extracts and prints basic statistics (mean, std deviation, skewness) for key DTI datasets (DAVIS, KIBA, BindingDB_Kd).
"""

import pandas as pd
import numpy as np
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log
from scipy.stats import skew

# Define datasets to analyze
datasets = {
    'DAVIS': DTI(name='DAVIS'),
    'KIBA': DTI(name='KIBA'),
    'BindingDB_Kd': DTI(name='BindingDB_Kd')
}

# Collect statistics
stats = []
for name, dataset in datasets.items():
    data = dataset.get_data()
    data['Y'] = convert_to_log(data['Y'])

    mean_val = data['Y'].mean()
    std_val = data['Y'].std()
    skew_val = skew(data['Y'])

    stats.append({
        'Dataset': name,
        'Mean (log)': round(mean_val, 3),
        'Std Dev': round(std_val, 3),
        'Skewness': round(skew_val, 3)
    })

# Convert to DataFrame and print
stats_df = pd.DataFrame(stats)
print("\n📊 Dataset Binding Affinity Statistics:")
print(stats_df)

# Optionally, save to CSV
stats_df.to_csv("results/dataset_statistics.csv", index=False)
print("\n✅ Statistics saved to results/dataset_statistics.csv")
