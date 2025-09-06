import argparse
import os
from os.path import join

import pandas as pd

from training import predict
from feature_utils import conbine_features, combine_ssim, combine_msssim
from train_model import get_features as get_vmaf_features
from feature_utils import get_ssim_features, get_msssim_features


def remove_extensions(file_name):
    """
    Remove file extensions and _4k suffix from file names
    """
    if file_name.endswith('.yuv'):
        return file_name[:-4]
    elif file_name.endswith('.mp4'):
        return file_name[:-4]
    elif file_name.endswith('_4k'):
        return file_name[:-3]
    else:
        return file_name


def main():
    argparser = argparse.ArgumentParser(
        description='Unified prediction tool for video quality assessment')
    argparser.add_argument('feature_path', type=str,
                           help='Path to the folder containing the features')
    argparser.add_argument('output_name', type=str,
                           help='Output name', default='predict.csv')
    argparser.add_argument(
        '--model', type=str, 
        help='Which model to use. Options: VMAF, SSIM, MSSSIM', 
        default='VMAF')
    args = argparser.parse_args()
    
    feats_pth = args.feature_path
    output_name = args.output_name
    model_type = args.model.upper()
    
    # Get features based on model type
    if model_type == 'VMAF':
        # Get VMAF features
        feature, nonlinear_features = get_vmaf_features(feats_pth)
        features = feature.merge(nonlinear_features, on='video')
        svr_path = 'models/svr/model_svr_livehdr.pkl'
        scaler_path = 'models/scaler/model_scaler_livehdr.pkl'
        print("VMAF model used")
    
    elif model_type == 'SSIM':
        # Get SSIM features
        ssim_features, nonlinear_features = get_ssim_features(feats_pth)
        features = ssim_features.merge(nonlinear_features, on='video')
        svr_path = 'models/svr/ssim_svr.pkl'
        scaler_path = 'models/scaler/ssim_scaler.pkl'
        print("SSIM model used")
    
    elif model_type == 'MSSSIM':
        # Get MS-SSIM features
        msssim_features, nonlinear_features = get_msssim_features(feats_pth)
        features = msssim_features.merge(nonlinear_features, on='video')
        svr_path = 'models/svr/msssim_svr.pkl'
        scaler_path = 'models/scaler/msssim_scaler.pkl'
        print("MS-SSIM model used")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: VMAF, SSIM, MSSSIM")
    
    # Convert column names to strings to avoid type issues
    features.columns = features.columns.astype(str)
    
    # Predict quality scores
    res = predict(features, svr_path, scaler_path)
    
    # Save results
    res.to_csv(output_name)
    print(f"Results saved to {output_name}")


if __name__ == "__main__":
    main()