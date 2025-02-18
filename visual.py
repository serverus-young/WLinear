import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'dataset/CoalCH4.csv'
df = pd.read_csv(file_path, parse_dates=[0], index_col=0)

downsampled_df = df.iloc[::10, :]

rolling_mean_df = df.rolling(window=24).mean()

fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(10, 3 * len(df.columns)), sharex=True)

for i, col in enumerate(df.columns):
    axes[i].plot(df.index, df[col], label=f'{col} (Original)', alpha=0.25, color = 'gray')

    axes[i].plot(downsampled_df.index, downsampled_df[col], label=f'{col} (Downsampled)', marker='o', linestyle='-',
                 markersize=2, color='blue')

    axes[i].plot(rolling_mean_df.index, rolling_mean_df[col], label=f'{col} (Rolling Mean)', linewidth=2, color='red')

    axes[i].set_title(col)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
# plt.show()
plt.savefig('dataset/CoalCH4.png')