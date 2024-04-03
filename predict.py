from training import *
import glob
import pandas as pd
import numpy as np
import sklearn
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from os.path import join

import sys
import re
import argparse


argparser = argparse.ArgumentParser(
    description='Predicting the video level features')
argparser.add_argument('feature_path', type=str,
                       help='Path to the folder containing the features')
argparser.add_argument('output_name', type=str,
                       help='Output name', default='predict.csv')
argparser.add_argument(
    '--model', type=str, help='Which model to use. Options: LIVEHDR, LIVEAQ, CUSTOM', default='LIVEHDR')
argparser.add_argument('--svr_name', type=str,
                       help='SVR name, to be used with CUSTOM model')
argparser.add_argument('--scaler_name', type=str,
                       help='Scaler name, to be used with CUSTOM model')
args = argparser.parse_args()
feats_pth = args.feature_path
output_name = args.output_name


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


def conbine_features(nonlinear, pth):

    if not nonlinear:
        dlm_nononlinear_feats_pth = join(
            feats_pth, f'hdrdlmnew/dlm_{pth}')
        dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
        vif_nononlinear_feats_pth = join(
            feats_pth, f'hdrvifnew/vif_{pth}')
        vif_feats_non = read_features(vif_nononlinear_feats_pth)

    else:
        dlm_nononlinear_feats_pth = join(
            feats_pth, f'hdrdlmnew/dlm_{pth}')
        dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
        vif_nononlinear_feats_pth = join(
            feats_pth, f'hdrvifnew/dlm_{pth}')
        vif_feats_non = read_features(vif_nononlinear_feats_pth)
        non_feats = vif_feats_non.merge(dlm_feats_non, on='video')

    return dlm_feats_non,vif_feats_non


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




dlm_feats,vif_feats = conbine_features( False,'none_2.0/ycbcr_0')
dlm_feats_nonlinear,vif_feats_nonlinear = conbine_features( False,'local_m_exp_None/ycbcr_0')


features = dlm_feats.merge(vif_feats, on="video", suffixes=('_dlm', '_vif'))
features = features.merge(dlm_feats_nonlinear, on="video", suffixes=('', '_dlm_nonlinear'))
features = features.merge(vif_feats_nonlinear, on="video", suffixes=('', '_vif_nonlinear'))


# feature = feature.merge(nonlinear_features_part1, on='video')
# feature = feature.merge(nonlinear_features_part2, on='video')
# print(feature.shape)
r = 0
# best = 0
if args.model.lower() == 'livehdr':
    res = predict(features, f'models/svr/model_svr_livehdr.pkl',
                    f'models/scaler/model_scaler_livehdr.pkl')
    print("hdrlive model used")
elif args.model.lower() == 'liveaq':
    res = predict(features, 'models/svr/model_svr_liveaq.pkl',
                    'models/scaler/model_scaler_liveaq.pkl')
    print("hdrqa model used")
elif args.model.lower() == 'custom':
    res = predict(features, args.svr_name, args.scaler_name)
    print("custom model used")
res.to_csv(output_name)
