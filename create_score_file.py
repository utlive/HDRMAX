import pandas as pd
import sys

try:
    # 读取分数文件
    print("Reading scores file...")
    scores_df = pd.read_csv('/mnt/data/videos/LIVE_HDR/scores/sureal_dark.csv')
    print(f"Successfully read scores file with {scores_df.shape[0]} rows and {scores_df.shape[1]} columns")

    # 读取视频信息文件
    print("Reading video info file...")
    video_info_df = pd.read_csv('/mnt/data/videos/LIVE_HDR/videos/video_info.csv')
    print(f"Successfully read video info file with {video_info_df.shape[0]} rows and {video_info_df.shape[1]} columns")

    # 检查两个文件中的video列格式
    print("\nScores DataFrame video examples:")
    print(scores_df['video'].head())
    print("\nVideo Info DataFrame video examples:")
    print(video_info_df['video'].head())

    # 处理video列格式
    # 在scores_df中，video列的格式是"4k_15M_Light2"
    # 在video_info_df中，video列的格式是"1080p_1M_Bonfire"，没有"_4k"后缀
    # 我们需要确保两个文件中的video列格式一致
    
    # 为video_info_df中的video列添加"_4k"后缀
    video_info_df['video'] = video_info_df['video'] + '_4k'
    
    print("\nUpdated Video Info DataFrame video examples:")
    print(video_info_df['video'].head())

    # 合并两个DataFrame
    print("\nMerging DataFrames...")
    merged_df = pd.merge(scores_df, video_info_df[['video', 'content']], on='video', how='inner')

    # 检查合并结果
    print("Merged DataFrame shape:", merged_df.shape)
    print("Merged DataFrame columns:", merged_df.columns.tolist())
    print("Number of rows in scores_df:", scores_df.shape[0])
    print("Number of rows in video_info_df:", video_info_df.shape[0])
    print("Number of rows in merged_df:", merged_df.shape[0])

    # 如果合并后的行数明显减少，可能是因为video列的格式不匹配
    if merged_df.shape[0] < min(scores_df.shape[0], video_info_df.shape[0]) * 0.5:
        print("\nWARNING: Significant reduction in rows after merging. Check video column formats.")
        
        # 检查哪些video在scores_df中但不在video_info_df中
        scores_videos = set(scores_df['video'])
        info_videos = set(video_info_df['video'])
        missing_videos = scores_videos - info_videos
        
        print(f"\n{len(missing_videos)} videos in scores_df but not in video_info_df:")
        for v in list(missing_videos)[:10]:  # 只显示前10个
            print(v)
            
        # 尝试不同的匹配方式
        print("\nTrying different matching approach...")
        # 从scores_df的video列中提取内容名称（如"Light2"）
        scores_df['content_name'] = scores_df['video'].str.split('_').str[-1]
        # 从video_info_df的content列中提取内容名称
        video_info_df['content_name'] = video_info_df['content']
        
        # 创建一个内容名称到内容的映射
        content_map = dict(zip(video_info_df['content_name'], video_info_df['content']))
        
        # 直接将内容映射到scores_df
        scores_df['content'] = scores_df['content_name'].map(content_map)
        
        # 使用scores_df作为合并结果
        merged_df = scores_df
        print("New merged DataFrame shape:", merged_df.shape)

    # 选择需要的列并重命名
    result_df = merged_df[['video', 'dark_dmos', 'content']]
    result_df = result_df.rename(columns={'dark_dmos': 'score'})

    # 保存到新文件
    result_df.to_csv('score_file.csv', index=False)

    print("\nScore file created successfully with", result_df.shape[0], "rows.")
    print("Sample of the score file:")
    print(result_df.head())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)