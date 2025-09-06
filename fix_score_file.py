import pandas as pd

# 读取分数文件
print("Reading score file...")
score_df = pd.read_csv('score_file.csv')
print(f"Original score file shape: {score_df.shape}")
print("Original video names examples:")
print(score_df['video'].head())

# 添加_4k后缀
score_df['video'] = score_df['video'] + '_4k'

print("\nUpdated video names examples:")
print(score_df['video'].head())

# 保存到新文件
score_df.to_csv('score_file_fixed.csv', index=False)
print("\nFixed score file created successfully.")