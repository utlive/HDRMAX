import glob
import os
import re
import numpy as np
import pandas as pd
from os.path import join

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
            if vname.endswith('_4k'):
                vname = vname[:-3]
            vnames.append(vname)
            feats.append(feats_1vid)
    else:
        for f in files:
            df = pd.read_csv(f, index_col=0)
            feats_1vid = process_vif(df)
            vname = os.path.basename(f)[:-4]
            if vname.endswith('_4k'):
                vname = vname[:-3]
            vnames.append(vname)
            feats.append(feats_1vid)
    features = pd.DataFrame(np.array(feats))
    features['video'] = vnames
    return features

def conbine_features(feats_pth, nonlinear=False, pth='none_2.0/ycbcr_0'):
    if nonlinear:
        pth = 'local_m_exp_None/ycbcr_0'
    
    dlm_feats_pth = join(feats_pth, f'hdrdlmnew/dlm_{pth}')
    dlm_feats = read_features(dlm_feats_pth)
    vif_feats_pth = join(feats_pth, f'hdrvifnew/vif_{pth}')
    vif_feats = read_features(vif_feats_pth)
    
    return dlm_feats, vif_feats

def combine_ssim(feats_pth):
    """
    Read SSIM features
    
    Args:
        feats_pth: Feature path
        
    Returns:
        DataFrame containing SSIM features
    """
    csv_files = glob.glob(join(feats_pth, 'hdrssimnew/*.csv'))
    vnames = []
    feats = []
    for f in csv_files:
        df = pd.read_csv(f)
        vname = os.path.basename(f)[:-4]
        if vname.endswith('_4k'):
            vname = vname[:-3]
        vnames.append(vname)
        # Read the whole row as features
        feats.append(df.values.reshape(-1))
    features = pd.DataFrame(np.array(feats))
    features['video'] = vnames
    return features

def combine_msssim(feats_pth):
    """
    Read MS-SSIM features
    
    Args:
        feats_pth: Feature path
        
    Returns:
        DataFrame containing MS-SSIM features
    """
    csv_files = glob.glob(join(feats_pth, 'hdrmsssimnew/*.csv'))
    vnames = []
    feats = []
    for f in csv_files:
        df = pd.read_csv(f)
        vname = os.path.basename(f)[:-4]
        if vname.endswith('_4k'):
            vname = vname[:-3]
        vnames.append(vname)
        # Read the whole row as features
        feats.append(df.values.reshape(-1))
    features = pd.DataFrame(np.array(feats))
    features['video'] = vnames
    return features

def get_ssim_features(feats_pth):
    """
    Get SSIM features and nonlinear features
    
    Args:
        feats_pth: Feature path
        
    Returns:
        ssim_features: SSIM features
        nonlinear_features: Nonlinear features
    """
    # Get SSIM features
    ssim_features = combine_ssim(feats_pth)
    
    # Get nonlinear features
    dlm_feats, vif_feats = conbine_features(feats_pth, True)
    nonlinear_features = vif_feats.merge(dlm_feats, on='video')
    
    return ssim_features, nonlinear_features

def get_msssim_features(feats_pth):
    """
    Get MS-SSIM features and nonlinear features
    
    Args:
        feats_pth: Feature path
        
    Returns:
        msssim_features: MS-SSIM features
        nonlinear_features: Nonlinear features
    """
    # Get MS-SSIM features
    msssim_features = combine_msssim(feats_pth)
    
    # Get nonlinear features
    dlm_feats, vif_feats = conbine_features(feats_pth, True)
    nonlinear_features = vif_feats.merge(dlm_feats, on='video')
    
    return msssim_features, nonlinear_features