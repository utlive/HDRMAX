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


def predict(df, svr_pth, scaler_pth):
    X_test = df.drop(['video'], axis=1)
    scaler = StandardScaler()
    svrfile = open(svr_pth, 'rb')
    grid_svr = pickle.load(svrfile)
    svrfile.close()
    scalerfile = open(scaler_pth, 'rb')
    scaler = pickle.load(scalerfile)
    scalerfile.close()
    X_test = scaler.transform(X_test)
    predict = grid_svr.predict(X_test)
    df['pred'] = predict
    return df


def combine_ssim(feats_pth):
    csv_files = glob.glob(join(feats_pth, 'hdrssimnew/*.csv'))
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
    description='Predicting the video level features')
argparser.add_argument('feature_path', type=str,
                       help='Path to the folder containing the features')
argparser.add_argument('output_name', type=str,
                       help='Output name', default='predict.csv')
argparser.add_argument('--svr_name', type=str,
                       help='SVR name, to be used with CUSTOM model',default="./models/svr/ssim_svr.pkl")
argparser.add_argument('--scaler_name', type=str,
                       help='Scaler name, to be used with CUSTOM model', default="./models/scaler/ssim_scaler.pkl")
args = argparser.parse_args()
output_name = args.output_name
feats_pth = args.feature_path
scaler_name = os.path.abspath(args.scaler_name)
svr_name = os.path.abspath(args.svr_name)
ssim_features = combine_ssim(feats_pth)
nonlinear_features = conbine_texp_features(feats_pth)
feature = ssim_features.merge(nonlinear_features, on='video')

res = predict(feature, args.svr_name, args.scaler_name)
res.to_csv(output_name)
