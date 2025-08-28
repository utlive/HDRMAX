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
import colour
import socket
import sys
from utils.dlm_utils import csf, dlm
import argparse
from datetime import datetime
import pdb
import warnings

def global_exp(image,par):

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

    avg_window = gen_gauss_window(patch_size//2, 7.0/6.0)
    mu_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode='constant')
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
    y = np.exp(par*(image - mu_image))
    return y
    

def m_exp(image):
    Y = image
    if(len(Y.shape) == 2):
        Y = np.expand_dims(Y, axis=2)
    
    maxY = scipy.ndimage.maximum_filter(Y, size=(17, 17, 1))
    minY = scipy.ndimage.minimum_filter(Y, size=(17, 17, 1))
    
    # Handle numerical stability as per original paper requirements
    diff = maxY - minY
    # Set Y_scaled to 0 where MAX=MIN or difference is too small (< 10^-3)
    mask = diff < 1e-3
    
    # Linear mapping from [minY, maxY] to [-1, 1]
    # MAX maps to +1, MIN maps to -1
    Y_scaled = -1 + (Y - minY) * 2 / diff
    
    # Set to 0 where diff is too small to avoid numerical issues
    Y_scaled[mask] = 0
    
    Y_transform = np.exp(np.abs(Y_scaled)*4)-1
    Y_transform[Y_scaled < 0] = -Y_transform[Y_scaled < 0]
    Y_transform = Y_transform.squeeze()
    return Y_transform

def global_m_exp(Y,delta):
    Y = -4+(Y-np.amin(Y))* 8/(1e-3+np.amax(Y)-np.amin(Y))
    Y_transform =  np.exp(np.abs(Y)**delta)-1
    Y_transform[Y<0] = -Y_transform[Y<0]
    return Y_transform
def logit(Y,par):
    
    maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
    minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
    delta = par
    Y_scaled = -0.99+1.98*(Y-minY)/(1e-3+maxY-minY)
    Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
    if(delta%2==0):
        Y_transform[Y<0] = -Y_transform[Y<0] 
    return Y_transform

def global_logit(Y,par):
    delta = par
    Y_scaled = -0.99+1.98*(Y-np.amin(Y))/(1e-3+np.amax(Y)-np.amin(Y))
    Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
    if(delta%2==0):
        Y_transform[Y<0] = -Y_transform[Y<0] 
    return Y_transform


def vif_refall_wrapper(ind):
    dis_f = files[ind]
    ref_f = ref_names[ind]
    print(dis_f,ref_f)
    dis_f = os.path.join(vid_pth,dis_f)
    ref_f = os.path.join(vid_pth,ref_f)
    vif_video_wrapper(ref_f,dis_f,ind)
    



def vif_video_wrapper(ref_f,dis_f,dis_index):
    

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    basename = os.path.basename(dis_f)
    print(basename)
    
    if(ref_f==dis_f):
        print('Videos are the same')
        return
    h = 2160 #hs[dis_index]
    w = 3840 #ws[dis_index]
    if args.frame_range == 'all':
        start = 0
        # get the number of frames using file size
        dis_num_frames = os.path.getsize(dis_f) // (h * w * 3)
        ref_num_frames = os.path.getsize(ref_f) // (h * w * 3)
        if dis_num_frames != ref_num_frames:
            # throw a warning
            warnings.warn('The number of frames in the reference and distorted videos are not the same. The smaller of the two will be used.')
        end = min(dis_num_frames, ref_num_frames)
    else:
        start = start_list[dis_index]
        end = end_list[dis_index]

            
        
    
    vif_image_wrapper(ref_f,dis_f,start,end,h,w,space = args.space, channel = args.channel, ind = dis_index,nonlinear = args.nonlinear, par = args.parameter, use_adaptive_csf=False,runvif = args.vif,rundlm = args.dlm)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
def vif_image_wrapper(ref_f,dis_f,start,end,h,w,space, channel ,ind, nonlinear = None,use_adaptive_csf=True,adaptation='bilateral',use_non_overlapping_blocks=True,use_views=False,par = None,runvif = False,rundlm = False):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    
    framelist =  list(range(start,end,int(fps[ind])))
    print(f'Extracting frames  from {start} to {end}')
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    if runvif:
        output_csv_vif = os.path.join(out_pth_vif,dis_name+'.csv')
        print('name to be is ',output_csv_vif)
        if((os.path.exists(output_csv_vif)==True) and (os.path.getsize(output_csv_vif) >100)):
            print(output_csv_vif,' is found')
            return
        f1 =  open(output_csv_vif,'w') 
        writer_vif=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer_vif.writerow(['framenum','vif','nums','denoms'])
    if rundlm:
        output_csv_dlm = os.path.join(out_pth_dlm,dis_name+'.csv')
        print('name to be is ',output_csv_dlm)
        if((os.path.exists(output_csv_dlm)==True) and (os.path.getsize(output_csv_dlm) >100)):
            print(output_csv_dlm,' is found')
            return
        f1 =  open(output_csv_dlm,'a') 
        writer_dlm=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer_dlm.writerow(['framenum','vif','nums','denoms'])
    for framenum in framelist:
        try:
            ref_multichannel = hdr_yuv_read(ref_file_object,framenum,h,w)
            dis_multichannel = hdr_yuv_read(dis_file_object,framenum,h,w)

        except Exception as e:
            print(e)
            break
        
        if (space == 'ycbcr'):
            if (nonlinear.lower()!='none'):
                ref_multichannel = [i.astype(np.float64)/1023 for i in ref_multichannel]
                dis_multichannel = [i.astype(np.float64)/1023 for i in dis_multichannel]
        elif(space == 'lab'):
            #first convert to 0-1 scale for the conversion
                ref_multichannel = np.stack(ref_multichannel,axis = 2)
                dis_multichannel = np.stack(dis_multichannel,axis = 2)
                ref_multichannel = ref_multichannel.astype(np.float64)/1023
                dis_multichannel = dis_multichannel.astype(np.float64)/1023
                frame = colour.YCbCr_to_RGB(ref_multichannel,K = [0.2627,0.0593])
                xyz = colour.RGB_to_XYZ(frame, [0.3127,0.3290], [0.3127,0.3290], 
                colour.models.RGB_COLOURSPACE_BT2020.RGB_to_XYZ_matrix, 
                chromatic_adaptation_transform='CAT02', 
                cctf_decoding=colour.models.eotf_PQ_BT2100)/10000
                lab = colour.XYZ_to_hdr_CIELab(xyz, illuminant=[ 0.3127, 0.329 ], Y_s=0.2, Y_abs=100, method='Fairchild 2011')
                ref_multichannel = lab
                frame = colour.YCbCr_to_RGB(dis_multichannel,K = [0.2627,0.0593])
                xyz = colour.RGB_to_XYZ(frame, [0.3127,0.3290], [0.3127,0.3290], 
                colour.models.RGB_COLOURSPACE_BT2020.RGB_to_XYZ_matrix, 
                chromatic_adaptation_transform='CAT02', 
                cctf_decoding=colour.models.eotf_PQ_BT2100)/10000
                lab = colour.XYZ_to_hdr_CIELab(xyz, illuminant=[ 0.3127, 0.329 ], Y_s=0.2, Y_abs=100, method='Fairchild 2011')
                dis_multichannel = lab
                ref_multichannel = ref_multichannel.transpose(2,0,1)
                dis_multichannel = dis_multichannel.transpose(2,0,1)
                if (nonlinear.lower()!='none'):
                    ref_multichannel /= 100
                    dis_multichannel /= 100

        ref_singlechannel = ref_multichannel[channel]
        dis_singlechannel = dis_multichannel[channel]       

        if(use_adaptive_csf==True):
            # apply CSF here
            if(use_non_overlapping_blocks==True): # apply CSF on non-overlapping blocks of the image
                csf_filtered_ref_y_pq = blockwise_csf(ref_singlechannel,adaptation=adaptation)
                csf_filtered_dis_y_pq = blockwise_csf(dis_singlechannel,adaptation=adaptation)
            else: # sliding window; returns filtered value at center of each sliding window
                csf_filtered_ref_y_pq = windows_csf(ref_singlechannel,use_views=use_views)
                csf_filtered_dis_y_pq = windows_csf(dis_singlechannel,use_views=use_views)

            # standard VIF but without CSF
            if runvif:
                vif_val = vif(csf_filtered_ref_y_pq,csf_filtered_dis_y_pq)
            if rundlm:
                dlm_val = dlm(csf_filtered_ref_y_pq,csf_filtered_dis_y_pq)
        elif(nonlinear == 'local_logit'):
            logit_ref = logit(ref_singlechannel,par)
            logit_dis = logit(dis_singlechannel,par)
            if runvif:
                vif_val = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val = dlm(logit_ref,logit_dis)
        elif(nonlinear == 'global_logit'):
            logit_ref = global_logit(ref_singlechannel,par)
            logit_dis = global_logit(dis_singlechannel,par)
            if runvif:
                vif_val = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val = dlm(logit_ref,logit_dis)
        elif(nonlinear == 'local_exp'):
            
            logit_ref = local_exp(ref_singlechannel,par,31)
            logit_dis = local_exp(dis_singlechannel,par,31)
            if runvif:
                vif_val = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val = dlm(logit_ref,logit_dis)
        elif(nonlinear == 'local_m_exp'):
            logit_ref = m_exp(ref_singlechannel)
            logit_dis = m_exp(dis_singlechannel)
            if runvif:
                vif_val = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val = dlm(logit_ref,logit_dis)   
        elif(nonlinear == 'global_m_exp'):
            logit_ref = global_m_exp(ref_singlechannel,par)
            logit_dis = global_m_exp(dis_singlechannel,par)
            if runvif:
                vif_val = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val = dlm(logit_ref,logit_dis)   
                               
        elif(nonlinear == 'none'):
            # standard VIF 
            if runvif:
                vif_val = vif(ref_singlechannel,dis_singlechannel)
            if rundlm:
                dlm_val = dlm(ref_singlechannel,dis_singlechannel)   
        # standard VIF
        if runvif:

            row = [framenum,vif_val[0],vif_val[1],vif_val[2]]
            writer_vif.writerow(row)
        if rundlm:

            row = [framenum,dlm_val[0],dlm_val[1],dlm_val[2]]
            writer_dlm.writerow(row)


parser = argparse.ArgumentParser()
parser.add_argument('vid_pth', type=str, help='directory containing reference data')
parser.add_argument('feature_path', type=str, help='directory containing distorted data')
parser.add_argument('csv_file_vidinfo', type=str, help='csv_file_vidinfo')
parser.add_argument("--space",help="choose which color space. Support 'ycbcr' and 'lab'.")
parser.add_argument("--nonlinear",help="select the nonliearity. Support 'global_logit','local_logit', 'local_m_exp','global_m_exp', 'local_exp', 'global_exp' or 'none'.")
parser.add_argument("--parameter",help="the parameter for the nonliear. Use with --nonliear",type=float)
parser.add_argument("--channel",help="indicate which channel to process. Please provide 0, 1, or 2",type=int)
parser.add_argument("--vif", help="obtaining vif output.", action="store_true")
parser.add_argument("--dlm", help="obtaining vif output.", action="store_true")
parser.add_argument("--njobs", help="Number of videos processed at the same time.")
parser.add_argument("--frame_range", type=str, default='all',
                    help="frame range to process. 'all' or 'file'. if 'all', the whole video is used to estimate the quality. if 'file', the video uses the 'start' and 'end' columns in the csv file to estimate the quality.")
args = parser.parse_args()

print(args.space)

csv_file_vidinfo = args.csv_file_vidinfo
vid_pth = args.vid_pth
feature_path = args.feature_path


out_root_vif = join(feature_path,'hdrvifnew/vif') 
out_root_dlm = join(feature_path,'hdrdlmnew/dlm') 
njobs = args.njobs
njobs = eval(njobs)
df_vidinfo = pd.read_csv(csv_file_vidinfo)
files = df_vidinfo["encoded_yuv"]
fps = df_vidinfo["fps"]
if args.frame_range == 'file':
    try:
        start_list = df_vidinfo["start"]
        end_list = df_vidinfo["end"]
    except:
        raise ValueError("Please provide 'start' and 'end' columns in the csv file when --frame_range is 'file'.")
    

ref_names = df_vidinfo["ref_yuv"]
if args.vif:
    out_pth_vif = f'{out_root_vif}_{args.nonlinear}_{args.parameter}/{args.space}_{args.channel}'
    if not os.path.exists(out_pth_vif):
        os.makedirs(out_pth_vif)

if args.dlm:
    out_pth_dlm = f'{out_root_dlm}_{args.nonlinear}_{args.parameter}/{args.space}_{args.channel}'
    if not os.path.exists(out_pth_dlm):
        os.makedirs(out_pth_dlm)

Parallel(n_jobs=njobs,verbose=1,backend="multiprocessing")(delayed(vif_refall_wrapper)(i) for i in range(len(files)))
