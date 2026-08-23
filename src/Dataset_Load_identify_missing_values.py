import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset" / "HR_Employee_Attrition_raw.csv"

df = pd.read_csv(dataset_path)

print("--- First 5 Rows ---")
print(df.head())
print("----print 6 columns----")
subset = df.iloc[:, 0:6]
print(subset)
missing_counts = df.isnull().sum()
total_missing_values = df.isnull().sum().sum()
print("-----Missing Values Per Column:----------")
print(missing_counts)
print(total_missing_values)
print("-" * 40)
duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
