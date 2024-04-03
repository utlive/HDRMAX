import argparse
import glob
import itertools
import os
import re
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
# from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from training import *


def remove_extensions(file_name):
    if file_name.endswith('.yuv'):
        return file_name[:-4]
    elif file_name.endswith('.mp4'):
        return file_name[:-4]
    else:
        return file_name


def process_string(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('   ', ' ')
    s = s.replace('  ', ' ')
    s = s.replace('\n', '')
    sp = re.split(' |,', s)
    sp = [value for value in sp if value != '']
    l = [eval(i) for i in sp]

    return l



def process_dlm(df, method='by_level'):

    # if the method is by band, then returns only four levels, otherwise returns the entire bands
    if method == 'by_level':

        nums2 = df.iloc[:, 1]
        nums2 = [process_string(i) for i in nums2]
        nums2 = np.array(nums2)
        nums2 = nums2.mean(axis=0)
        return nums2
    else:

        nums3 = df.iloc[:, 2]
        nums3 = [process_string(i) for i in nums3]
        nums3 = np.array(nums3)
        nums3 = nums3.mean(axis=0)
        return nums3


def process_vif(df, method='by_level'):

    # if the method is by band, then returns only four levels, otherwise returns the entire bands

    nums2 = df.iloc[:, 1]
    nums2 = [process_string(i) for i in nums2]
    nums2 = np.array(nums2)
    nums2 = nums2

    nums3 = df.iloc[:, 2]
    nums3 = [process_string(i) for i in nums3]
    nums3 = np.array(nums3)
    res = nums2.reshape(-1, 4, 2).sum(axis=2) / \
        (nums3.reshape(-1, 4, 2).sum(axis=2)+0.0001)
    res = res.mean(axis=0)
    return res


def read_features(file_path):
    files = glob.glob(join(file_path, '*.csv'))
    vnames = []
    feats = []
    if files == []:
        print('em')
    if file_path.find('dlm') >= 0:

        for f in files:
            df = pd.read_csv(f, index_col=0)

            feats_1vid = process_dlm(df)
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)
    else:
        for f in files:
            df = pd.read_csv(f, index_col=0)

            feats_1vid = process_vif(df)
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)
    features = pd.DataFrame(np.array(feats))
    features['video'] = vnames
    return features


def conbine_texp_features(feats_pth):
    bright_pth = join(feats_pth, f'hdrdlmnew/dlm_local_m_exp_None/ycbcr_0')
    dlm_feats_non = read_two_exp_features(bright_pth)
    bright_pth = join(feats_pth, f'hdrvifnew/vif_local_m_exp_None/ycbcr_0')
    vif_feats_non = read_two_exp_features(bright_pth)
    non_feats = vif_feats_non.merge(dlm_feats_non, on='video')
    return non_feats


def read_two_exp_features(pth1):
    files = glob.glob(join(pth1, '*.csv'))
    vnames = []
    feats = []
    if files == []:
        print('em')
    if pth1.find('dlm') >= 0:

        for f in files:
            df = pd.read_csv(f, skiprows=1, header=None, index_col=0)

            feats_1vid = process_dlm(df.iloc[:, [0, 1, 2]])
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)
    else:
        for f in files:
            df = pd.read_csv(f, skiprows=1, header=None, index_col=0)

            feats_1vid = process_vif(df.iloc[:, [0, 1, 2]])
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)

    features1 = pd.DataFrame(np.array(feats))
    features1['video'] = vnames

    return features1


def combine_ssim(feats_pth):
    csv_files = glob.glob(join(feats_pth, 'hdrmsssimnew/*.csv'))
    vnames = []
    feats = []
    for f in csv_files:
        df = pd.read_csv(f)
        vname = os.path.basename(f)[:-4]
        vnames.append(vname)
        # read the whole row as features
        feats.append(df.values.reshape(-1))
    features = pd.DataFrame(np.array(feats))
    features['video'] = vnames
    return features

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
scores['content'] = scores['video'].apply(lambda x: x.split('_')[2])

configs = []
ssim_features = combine_ssim(feats_pth)
nonlinear_features = conbine_texp_features(feats_pth)
feature = ssim_features.merge(nonlinear_features, on='video')
feature = feature.merge(scores[['video', 'score', 'content']], on='video')
train_for_srocc_svr(feature, scaler_name, svr_name)
print('Training finished. ')
