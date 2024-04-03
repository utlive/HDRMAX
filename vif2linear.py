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
    

def m_exp(image,par,patch_size = 31):
    maxY = scipy.ndimage.maximum_filter(image,size=(patch_size,patch_size))
    minY = scipy.ndimage.minimum_filter(image,size=(patch_size,patch_size))
    image = -4+(image-minY)* 8/(1e-3+maxY-minY)
    Y_transform =  np.exp(np.abs(image)**par)-1
    Y_transform[image<0] = -Y_transform[image<0]
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
    dis_f = upscaled_yuv_names[ind]
    content = os.path.basename(dis_f).split('_')[2]
    ref_f = '4k_ref_'+content + '_upscaled.yuv'
    print(dis_f,ref_f)
    dis_f = os.path.join(vid_pth,dis_f)
    ref_f = os.path.join(vid_pth,ref_f)
    vif_video_wrapper(ref_f,dis_f)
    



def vif_video_wrapper(ref_f,dis_f):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    basename = os.path.basename(dis_f)
    print(basename)
    if(ref_f==dis_f):
        print('Videos are the same')
        return
    dis_index = upscaled_yuv_names.index(basename)
    h = 2160 #hs[dis_index]
    w = 3840 #ws[dis_index]
    framenos = framenos_list[dis_index]
    vif_image_wrapper(ref_f,dis_f,framenos,h,w,space = args.space, channel = args.channel, nonlinear = args.nonlinear, par = args.parameter, use_adaptive_csf=False,runvif = args.vif,rundlm = args.dlm)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
def vif_image_wrapper(ref_f,dis_f,framenos,h,w,space, channel , nonlinear = None,use_adaptive_csf=True,adaptation='bilateral',use_non_overlapping_blocks=True,use_views=False,par = None,runvif = False,rundlm = False):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    framelist =  list(range(0,framenos,50))

    score_df = pd.DataFrame([])
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    if runvif:
        output_csv_vif = os.path.join(out_pth_vif,dis_name+'.csv')
        print('name to be is ',output_csv_vif)
        if((os.path.exists(output_csv_vif)==True) and (os.path.getsize(output_csv_vif) >100)):
            print(output_csv_vif,' is found')
            return
        f1 =  open(output_csv_vif,'a') 
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
        ref_singlechannel = colour.models.eotf_PQ_BT2100(ref_singlechannel)/10000
        dis_singlechannel = colour.models.eotf_PQ_BT2100(dis_singlechannel)/10000
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
                vif_val1 = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val1 = dlm(logit_ref,logit_dis)
            logit_ref = local_exp(ref_singlechannel,-par,31)
            logit_dis = local_exp(dis_singlechannel,-par,31)
            if runvif:
                vif_val2 = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val2 = dlm(logit_ref,logit_dis)
        elif(nonlinear == 'global_exp'):
            logit_ref = global_exp(ref_singlechannel,par)
            logit_dis = global_exp(dis_singlechannel,par)
            if runvif:
                vif_val1 = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val1 = dlm(logit_ref,logit_dis)
            logit_ref = global_exp(ref_singlechannel,-par)
            logit_dis = global_exp(dis_singlechannel,-par)
            if runvif:
                vif_val2 = vif(logit_ref,logit_dis)
            if rundlm:
                dlm_val2= dlm(logit_ref,logit_dis)
        elif(nonlinear == 'local_m_exp'):
            logit_ref = m_exp(ref_singlechannel,par)
            logit_dis = m_exp(dis_singlechannel,par)
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
            if 'vif_val1' in locals():
                row = [framenum,vif_val1[0],vif_val1[1],vif_val1[2],vif_val2[0],vif_val2[1],vif_val2[2]]
            else:
                row = [framenum,vif_val[0],vif_val[1],vif_val[2]]
            writer_vif.writerow(row)
        if rundlm:
            if 'dlm_val1' in locals():
                row = [framenum,dlm_val1[0],dlm_val1[1],dlm_val1[2],dlm_val2[0],dlm_val2[1],dlm_val2[2]]
            else:
                row = [framenum,dlm_val[0],dlm_val[1],dlm_val[2]]
            writer_dlm.writerow(row)


parser = argparse.ArgumentParser()
parser.add_argument("--space",help="choose which color space. Support 'ycbcr' and 'lab'.")
parser.add_argument("--nonlinear",help="select the nonliearity. Support 'logit', 'm_exp', 'local_exp', 'global_exp' or 'none'.")
parser.add_argument("--parameter",help="the parameter for the nonliear. Use with --nonliear",type=float)
parser.add_argument("--channel",help="indicate which channel to process. Please provide 0, 1, or 2",type=int)
parser.add_argument("--vif", help="obtaining vif output.", action="store_true")
parser.add_argument("--dlm", help="obtaining vif output.", action="store_true")
args = parser.parse_args()


print(args.space)


if socket.gethostname().find('tacc')>0:
    scratch =  os.environ['SCRATCH']
    csv_file_vidinfo = 'fall2021_yuv_rw_info.csv'
    vid_pth = '/scratch/06776/kmd1995/video/HDR_2021_fall_yuv_upscaled/fall2021_hdr_upscaled_yuv'
    out_root_vif = join(scratch,'feats/feats_linear/hdrvifnew/vif')
    out_root_dlm = join(scratch,'feats/feats_linear/hdrdlmnew/dlm')

elif socket.gethostname().find('a51969')>0: #Odin
    csv_file_vidinfo = '/home/zaixi/code/HDRproject/hdr_vmaf/python_vmaf/fall2021_yuv_rw_info.csv'
    vid_pth = '/mnt/7e60dcd9-907d-428e-970c-b7acf5c8636a/fall2021_hdr_upscaled_yuv/'
    out_root_vif = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvifnew/vif'
    out_root_dlm = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrdlmnew/dlm'

elif socket.gethostname().find('895095')>0: #DarthVader
    csv_file_vidinfo = 'fall2021_yuv_rw_info.csv'
    vid_pth = '/media/josh/seagate/hdr_videos/fall2021_hdr_upscaled_yuv/'
    out_root_vif = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrvifnew/vif'
    out_root_dlm = '/media/zaixi/zaixi_nas/HDRproject/feats/hdrdlmnew/dlm'


df_vidinfo = pd.read_csv(csv_file_vidinfo)
files = df_vidinfo["yuv"]
ref_files = glob.glob(join(vid_pth,'4k_ref_*'))
fps = df_vidinfo["fps"]
framenos_list = df_vidinfo["framenos"] 
ws =df_vidinfo["w"]
hs = df_vidinfo["h"]
upscaled_yuv_names = [x[:-4]+'_upscaled.yuv' for x in df_vidinfo['yuv']]

if args.vif:
    out_pth_vif = f'{out_root_vif}_{args.nonlinear}_{args.parameter}/{args.space}_{args.channel}'
    if not os.path.exists(out_pth_vif):
        os.makedirs(out_pth_vif)

if args.dlm:
    out_pth_dlm = f'{out_root_dlm}_{args.nonlinear}_{args.parameter}/{args.space}_{args.channel}'
    if not os.path.exists(out_pth_dlm):
        os.makedirs(out_pth_dlm)


Parallel(n_jobs=31,verbose=1,backend="multiprocessing")(delayed(vif_refall_wrapper)(i) for i in range(len(files)))
