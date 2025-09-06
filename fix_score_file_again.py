import pandas as pd
import re

# 读取分数文件
print("Reading score file...")
score_df = pd.read_csv('score_file_fixed.csv')
print(f"Original score file shape: {score_df.shape}")
print("Original video names examples:")
print(score_df['video'].head())

# 移除_4k后缀
score_df['video'] = score_df['video'].str.replace('_4k', '')

print("\nUpdated video names examples:")
print(score_df['video'].head())

# 保存到新文件
score_df.to_csv('score_file_final.csv', index=False)
print("\nFinal score file created successfully.")