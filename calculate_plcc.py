import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import argparse

def calculate_plcc(pred_file, score_file):
    """
    Calculate Pearson Linear Correlation Coefficient (PLCC) between predicted scores and ground truth scores
    
    Args:
        pred_file: Path to the prediction file
        score_file: Path to the ground truth score file
        
    Returns:
        PLCC value
    """
    # Read prediction file
    pred_df = pd.read_csv(pred_file)
    
    # Read ground truth score file
    score_df = pd.read_csv(score_file)
    
    # Process video names to ensure they match
    pred_df['video'] = pred_df['video'].apply(lambda x: x.split('_4k')[0] if isinstance(x, str) and x.endswith('_4k') else x)
    score_df['video'] = score_df['video'].apply(lambda x: x.split('_4k')[0] if isinstance(x, str) and x.endswith('_4k') else x)
    
    # Merge dataframes on video column
    merged_df = pred_df.merge(score_df, on='video', how='inner')
    
    if merged_df.empty:
        print(f"No matching videos found between prediction file and score file!")
        return None
    
    # Calculate PLCC
    plcc, p_value = pearsonr(merged_df['pred'], merged_df['score'])
    
    print(f"Number of matched videos: {len(merged_df)}")
    print(f"PLCC: {plcc:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    return plcc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate PLCC between predicted scores and ground truth scores')
    parser.add_argument('pred_file', type=str, help='Path to the prediction file')
    parser.add_argument('score_file', type=str, help='Path to the ground truth score file')
    
    args = parser.parse_args()
    
    calculate_plcc(args.pred_file, args.score_file)