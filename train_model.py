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


def conbine_features(config, nonlinear):
    s, c, n, p = config
    if not nonlinear:
        dlm_nononlinear_feats_pth = join(
            feats_pth, f'hdrdlmnew/dlm_none_2.0/{s}_0')
        dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
        vif_nononlinear_feats_pth = join(
            feats_pth, f'hdrvifnew/vif_none_2.0/{s}_0')
        vif_feats_non = read_features(vif_nononlinear_feats_pth)
        non_feats = vif_feats_non.merge(dlm_feats_non, on='video')
        if c:
            dlm_nononlinear_feats_pth = join(
                feats_pth, f'hdrdlmnew/dlm_none_2.0/{s}_1')
            dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
            vif_nononlinear_feats_pth = join(
                feats_pth, f'hdrvifnew/vif_none_2.0/{s}_1')
            vif_feats_non = read_features(vif_nononlinear_feats_pth)
            non_feats1 = vif_feats_non.merge(dlm_feats_non, on='video')
            non_feats = non_feats.merge(non_feats1, on='video')
            dlm_nononlinear_feats_pth = join(
                feats_pth, f'hdrdlmnew/dlm_none_2.0/{s}_2')
            dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
            vif_nononlinear_feats_pth = join(
                feats_pth, f'hdrvifnew/vif_none_2.0/{s}_2')
            vif_feats_non = read_features(vif_nononlinear_feats_pth)
            non_feats1 = vif_feats_non.merge(dlm_feats_non, on='video')
            non_feats = non_feats.merge(non_feats1, on='video')
    else:
        dlm_nononlinear_feats_pth = join(
            feats_pth, f'hdrdlmnew/dlm_{n}_{p}/{s}_0')
        dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
        vif_nononlinear_feats_pth = join(
            feats_pth, f'hdrvifnew/vif_{n}_{p}/{s}_0')
        vif_feats_non = read_features(vif_nononlinear_feats_pth)
        non_feats = vif_feats_non.merge(dlm_feats_non, on='video')
        if c:
            dlm_nononlinear_feats_pth = join(
                feats_pth, f'hdrdlmnew/dlm_{n}_{p}/{s}_1')
            dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
            vif_nononlinear_feats_pth = join(
                feats_pth, f'hdrvifnew/vif_{n}_{p}/{s}_1')
            vif_feats_non = read_features(vif_nononlinear_feats_pth)
            non_feats1 = vif_feats_non.merge(dlm_feats_non, on='video')
            non_feats = non_feats.merge(non_feats1, on='video')
            dlm_nononlinear_feats_pth = join(
                feats_pth, f'hdrdlmnew/dlm_{n}_{p}/{s}_2')
            dlm_feats_non = read_features(dlm_nononlinear_feats_pth)
            vif_nononlinear_feats_pth = join(
                feats_pth, f'hdrvifnew/vif_{n}_{p}/{s}_2')
            vif_feats_non = read_features(vif_nononlinear_feats_pth)
            non_feats1 = vif_feats_non.merge(dlm_feats_non, on='video')
            non_feats = non_feats.merge(non_feats1, on='video')
    return non_feats


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


def conbine_texp_features(config):
    s, c, n, p1, p2 = config
    bright_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p1}/{s}_0')
    dark_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p2}/{s}_0')
    dlm_feats_non = read_two_exp_features(bright_pth, dark_pth)
    bright_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p1}/{s}_0')
    dark_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p2}/{s}_0')
    vif_feats_non = read_two_exp_features(bright_pth, bright_pth)
    non_feats = vif_feats_non.merge(dlm_feats_non, on='video')
    if c:
        bright_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p1}/{s}_1')
        dark_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p2}/{s}_1')
        dlm_feats_non = read_two_exp_features(bright_pth, dark_pth)
        bright_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p1}/{s}_1')
        dark_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p2}/{s}_1')
        vif_feats_non = read_two_exp_features(bright_pth, bright_pth)
        non_feats1 = vif_feats_non.merge(dlm_feats_non, on='video')
        non_feats = non_feats.merge(non_feats1, on='video')

        bright_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p1}/{s}_2')
        dark_pth = join(feats_pth, f'hdrdlmnew/dlm_{n}_{p2}/{s}_2')
        dlm_feats_non = read_two_exp_features(bright_pth, dark_pth)
        bright_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p1}/{s}_2')
        dark_pth = join(feats_pth, f'hdrvifnew/vif_{n}_{p2}/{s}_2')
        vif_feats_non = read_two_exp_features(bright_pth, bright_pth)
        non_feats2 = vif_feats_non.merge(dlm_feats_non, on='video')
        non_feats = non_feats.merge(non_feats2, on='video')
    return non_feats


def read_two_exp_features(pth1, pth2):
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

    files = glob.glob(join(pth2, '*.csv'))
    vnames = []
    feats = []
    if files == []:
        print('em')
    if pth1.find('dlm') >= 0:

        for f in files:
            df = pd.read_csv(f, skiprows=1, header=None, index_col=0)

            feats_1vid = process_dlm(df.iloc[:, [3, 4, 5]])
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)
    else:
        for f in files:
            df = pd.read_csv(f, skiprows=1, header=None, index_col=0)
            feats_1vid = process_vif(df.iloc[:, [3, 4, 5]])
            vname = os.path.basename(f)[:-4]
            vnames.append(vname)
            feats.append(feats_1vid)

    features2 = pd.DataFrame(np.array(feats))
    features2['video'] = vnames

    features = features1.merge(features2, on='video')

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


spaces = ['ycbcr']
nonlinear = ['local_exp']
counter = 0

configs = []

for s in spaces:
    for n in nonlinear:
        for p1 in [0.5]:
            for p2 in [5.0]:
                for c in [False]:
                    configs.append([s, c, n, p1, p2])

for cfig_index in range(0, len(configs)):
    print(configs[cfig_index])
    s, c, n, p1, p2 = configs[cfig_index]
    feature = conbine_features(configs[cfig_index][:-1], False)
    nonlinear_features = conbine_texp_features(configs[cfig_index])
    feature = feature.merge(nonlinear_features, on='video')
    feature = feature.merge(scores[['video', 'score', 'content']], on='video')
    train_for_srocc_svr(feature, scaler_name, svr_name)
    print('Training finished. ')
