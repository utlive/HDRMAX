from skimage.util.shape import view_as_blocks
import csv
from skimage import filters
import matplotlib.pyplot as plt
from utils.vif.vif_utils import vif
from joblib import dump,Parallel,delayed
from scipy.stats import gmean
import time
from scipy.ndimage import gaussian_filter
from utils.hdr_utils import hdr_yuv_read
from utils.csf_utils import csf_barten_frequency,csf_filter_block,blockwise_csf,windows_csf
import numpy as np
import glob
import pandas as pd
import os
from os.path import  join
import scipy

import socket
import sys


if socket.gethostname().find('tacc')>0:
        
    csv_file_vidinfo = 'fall2021_yuv_rw_info.csv'
    vid_pth = '/scratch/06776/kmd1995/video/HDR_2021_fall_yuv_upscaled/fall2021_hdr_upscaled_yuv'
    out_root = '/scratch/06776/kmd1995/feats/feats/hdrvif/vif'

else:
    csv_file_vidinfo = '/home/zaixi/code/HDRproject/hdr_vmaf/python_vmaf/fall2021_yuv_rw_info.csv'
    vid_pth = '/mnt/7e60dcd9-907d-428e-970c-b7acf5c8636a/fall2021_hdr_upscaled_yuv/'
    out_root = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvif/vif'


df_vidinfo = pd.read_csv(csv_file_vidinfo)
files = df_vidinfo["yuv"]
ref_files = glob.glob(join(vid_pth,'4k_ref_*'))
fps = df_vidinfo["fps"]
framenos_list = df_vidinfo["framenos"] //3
ws =df_vidinfo["w"]
hs = df_vidinfo["h"]
upscaled_yuv_names = [x[:-4]+'_upscaled.yuv' for x in df_vidinfo['yuv']]


def global_exp(image,par):
    if np.max(image) > 1.1:
        image = image/1023
    assert len(np.shape(image)) == 2
    avg = np.average(image)
    y = np.exp(par*(image-avg))
    return y

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def local_exp(image,par,patch_size):
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    if np.max(image) > 1.1:
        image = image/1023
    avg_window = gen_gauss_window(patch_size//2, 7.0/6.0)
    mu_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode='constant')
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
    y = np.exp(par*(image - mu_image))
    return y
    

def logit(Y,delta):
    Y = -0.99+(Y-np.amin(Y))* 1.98/(np.amax(Y)-np.amin(Y))
    Y_transform = np.log((1+(Y)**delta)/(1-(Y)**delta))
    return Y_transform

def vif_refall_wrapper(ind,files):
    ref_f = files[ind]
    content = os.path.basename(ref_f).split('_')[2]
    print(content)
    dis_filenames = [x for x in glob.glob(join(vid_pth,"*")) if content in x]
    print(dis_filenames)
    Parallel(n_jobs=31,verbose=1)(delayed(vif_video_wrapper)(ref_f,dis_f) for dis_f in dis_filenames)
#    for dis_f in dis_filenames:
#        vif_video_wrapper(ref_f,dis_f)
#    dis_f = dis_filenames[0]
#    vif_video_wrapper(ref_f,dis_f)

def vif_video_wrapper(ref_f,dis_f):
    basename = os.path.basename(dis_f)
    print(basename)
    if(ref_f==dis_f):
        print('Videos are the same')
        return
    dis_index = upscaled_yuv_names.index(basename)
    h = 2160 #hs[dis_index]
    w = 3840 #ws[dis_index]
    framenos = framenos_list[dis_index]
    vif_image_wrapper(ref_f,dis_f,framenos,h,w,nonlinear = nonlinear,par = par,use_adaptive_csf=False)

def vif_image_wrapper(ref_f,dis_f,framenos,h,w,nonlinear = None,use_adaptive_csf=True,adaptation='bilateral',use_non_overlapping_blocks=True,use_views=False,par = None):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    randlist =  np.random.randint(0,framenos,10)

    score_df = pd.DataFrame([])
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    output_csv = os.path.join(out_pth,dis_name+'.csv')
    print('name to be is ',output_csv)
    if((os.path.exists(output_csv)==True) and (os.path.getsize(output_csv) >1000)):
        print(output_csv,' is found')
        return
    with open(output_csv,'a') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['framenum','vif','nums','denoms'])
        for framenum in randlist:
            try:
                ref_y_pq,_,_ = hdr_yuv_read(ref_file_object,framenum,h,w)
                dis_y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)
            except Exception as e:
                print(e)
                break
            if(use_adaptive_csf==True):
                # apply CSF here
                if(use_non_overlapping_blocks==True): # apply CSF on non-overlapping blocks of the image
                    csf_filtered_ref_y_pq = blockwise_csf(ref_y_pq,adaptation=adaptation)
                    csf_filtered_dis_y_pq = blockwise_csf(dis_y_pq,adaptation=adaptation)
                else: # sliding window; returns filtered value at center of each sliding window
                    csf_filtered_ref_y_pq = windows_csf(ref_y_pq,use_views=use_views)
                    csf_filtered_dis_y_pq = windows_csf(dis_y_pq,use_views=use_views)

                # standard VIF but without CSF
                vif_val = vif(csf_filtered_ref_y_pq,csf_filtered_dis_y_pq)
            elif(nonlinear == 'logit'):
                logit_ref_y_pq = logit(ref_y_pq,1)
                logit_dis_y_pq = logit(dis_y_pq,1)
                vif_val = vif(logit_ref_y_pq,logit_dis_y_pq)
            elif(nonlinear == 'local_exp'):
                logit_ref_y_pq = local_exp(ref_y_pq,par,31)
                logit_dis_y_pq = local_exp(dis_y_pq,par,31)
                vif_val1 = vif(logit_ref_y_pq,logit_dis_y_pq)
                logit_ref_y_pq = local_exp(ref_y_pq,-par,31)
                logit_dis_y_pq = local_exp(dis_y_pq,-par,31)
                vif_val2 = vif(logit_ref_y_pq,logit_dis_y_pq)

            elif(nonlinear == 'global_exp'):
                logit_ref_y_pq = global_exp(ref_y_pq,par)
                logit_dis_y_pq = global_exp(dis_y_pq,par)
                vif_val1 = vif(logit_ref_y_pq,logit_dis_y_pq)
                logit_ref_y_pq = global_exp(ref_y_pq,-par)
                logit_dis_y_pq = global_exp(dis_y_pq,-par)
                vif_val2 = vif(logit_ref_y_pq,logit_dis_y_pq)
            else:
                # standard VIF 
                vif_val = vif(ref_y_pq,dis_y_pq)
            # standard VIF
            if vif_val1 is not 'None':
                row = [framenum,vif_val1[0],vif_val1[1],vif_val1[2],vif_val2[0],vif_val2[1],vif_val2[2]]
            else:
                row = [framenum,vif_val[0],vif_val[1],vif_val[2]]
            writer.writerow(row)



for nonlinear in ['local_exp','global_exp']:
    for par in [float(sys.argv[1])]:
        out_pth = f'{out_root}_{nonlinear}_{par}'

        if not os.path.exists(out_pth):
            os.makedirs(out_pth)
        Parallel(n_jobs=3,verbose=1)(delayed(vif_refall_wrapper)(i,ref_files) for i in range(len(ref_files)))