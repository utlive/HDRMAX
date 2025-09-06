import argparse
import os
from os.path import join

import pandas as pd

from training import train_for_srocc_svr
from feature_utils import get_ssim_features


def remove_extensions(file_name):
    if file_name.endswith('.yuv'):
        return file_name[:-4]
    elif file_name.endswith('.mp4'):
        return file_name[:-4]
    elif file_name.endswith('_4k'):
        return file_name[:-3]
    else:
        return file_name


# Main function to get features
def get_features(feats_pth):
    # Get SSIM features and nonlinear features
    ssim_features, nonlinear_features = get_ssim_features(feats_pth)
    
    return ssim_features, nonlinear_features


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Train a model on a custom dataset')
    argparser.add_argument('feature_path', type=str,
                        help='Path to the folder containing the features')
    argparser.add_argument('score_csv', type=str,
                        help='Path to the score file. This should be a csv file with the following columns: video, score, content. It is critical that the video column is the same as the video name in the feature folder.')
    argparser.add_argument('--scaler_name', type=str,
                        help='The name of the Scaler', default='model_scaler.pkl')
    argparser.add_argument('--svr_name', type=str,
                        help='The name of the SVR', default='model_svr.pkl')

    args = argparser.parse_args()
    feats_pth = args.feature_path
    score_csv = args.score_csv
    scaler_name = os.path.abspath(args.scaler_name)
    svr_name = os.path.abspath(args.svr_name)

    # create the folder to save the scaler if it does not exist.
    if not os.path.exists(os.path.dirname(scaler_name)):
        os.makedirs(os.path.dirname(scaler_name))

    # create the folder to save the svr if it does not exist.
    if not os.path.exists(os.path.dirname(svr_name)):
        os.makedirs(os.path.dirname(svr_name))

    # read the score file
    scores = pd.read_csv(score_csv)
    scores['video'] = scores['video'].apply(remove_extensions)
    
    # Get features
    ssim_features, nonlinear_features = get_features(feats_pth)

    # Merge features and scores
    feature = ssim_features.merge(nonlinear_features, on='video')
    feature = feature.merge(scores[['video', 'score', 'content']], on='video')

    # Train model
    train_for_srocc_svr(feature, scaler_name, svr_name)
    print('Training finished. ')
