from skimage.util.shape import view_as_blocks
from skimage import filters
import matplotlib.pyplot as plt
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
import argparse
from datetime import datetime
import warnings
from ssim_features import structural_similarity_features as ssim_features

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


def ssim_refall_wrapper(ind):
    dis_f = files[ind]
    ref_f = ref_names[ind]
    print(dis_f,ref_f)
    dis_f = os.path.join(vid_pth,dis_f)
    ref_f = os.path.join(vid_pth,ref_f)
    ssim_video_wrapper(ref_f,dis_f,ind)
    



def ssim_video_wrapper(ref_f,dis_f,dis_index):
    

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

            
        
    
    ssim_image_wrapper(ref_f,dis_f,start,end,h,w,space = args.space, channel = args.channel, ind = dis_index)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
def ssim_image_wrapper(ref_f,dis_f,start,end,h,w,space, channel ,ind):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    
    framelist =  list(range(start,end,int(fps[ind])))
    print(f'Extracting frames  from {start} to {end}')
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    output_csv_ssim = os.path.join(out_pth_ssim, dis_name+'.csv')
    ssim_feats = []
    for framenum in framelist:
        try:
            ref_multichannel = hdr_yuv_read(ref_file_object,framenum,h,w)
            dis_multichannel = hdr_yuv_read(dis_file_object,framenum,h,w)

        except Exception as e:
            print(e)
            break
        
        if (space == 'ycbcr'):
           
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
         

        ref_singlechannel = ref_multichannel[channel]
        dis_singlechannel = dis_multichannel[channel]       
        ssim_feat = ssim_features(ref_singlechannel, dis_singlechannel)
        ssim_feats.append(ssim_feat)
    ssim_feats = np.array(ssim_feats)
    # average over the frames
    ssim_feats = np.mean(ssim_feats, axis=0)
    # create a dataframe and save it
    df = pd.DataFrame(ssim_feats.reshape(1, -1))
    df.to_csv(output_csv_ssim, index=False)



    


parser = argparse.ArgumentParser()
parser.add_argument('vid_pth', type=str, help='directory containing reference data')
parser.add_argument('feature_path', type=str, help='directory containing distorted data')
parser.add_argument('csv_file_vidinfo', type=str, help='csv_file_vidinfo')
parser.add_argument("--space",help="choose which color space. Support 'ycbcr' and 'lab'.")
parser.add_argument("--channel",help="indicate which channel to process. Please provide 0, 1, or 2",type=int)
parser.add_argument("--njobs", help="Number of videos processed at the same time.",type=int,default=1)
parser.add_argument("--frame_range", type=str, default='all',
                    help="frame range to process. 'all' or 'file'. if 'all', the whole video is used to estimate the quality. if 'file', the video uses the 'start' and 'end' columns in the csv file to estimate the quality.")
args = parser.parse_args()

print(args.space)

csv_file_vidinfo = args.csv_file_vidinfo
vid_pth = args.vid_pth
feature_path = args.feature_path
njobs = args.njobs
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
out_pth_ssim = join(feature_path,'hdrssimnew')
os.makedirs(out_pth_ssim, exist_ok=True)

Parallel(n_jobs=njobs,verbose=1,backend="multiprocessing")(delayed(ssim_refall_wrapper)(i) for i in range(len(files)))
