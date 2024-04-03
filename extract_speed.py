import numpy as np
import os
import glob
import cv2
from joblib import Parallel,delayed,dump
import scipy.ndimage
import skimage.util
import math

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

def compute_speed(ref, ref_next, dis, dis_next, \
                             window):
    blk = 5;
    sigma_nsq = 0.1;
    times_to_down_size = 4; 

    #resize all frames
    for i in range(times_to_down_size):
        ref = np.array(cv2.resize(ref, None, fx=0.5, fy=0.5, \
                         interpolation=cv2.INTER_AREA),dtype=np.float32)
        ref_next = np.array(cv2.resize(ref_next, None, fx=0.5, fy=0.5, \
                              interpolation=cv2.INTER_AREA),dtype=np.float32)
        dis = np.array(cv2.resize(dis, None, fx=0.5, fy=0.5, \
                         interpolation=cv2.INTER_AREA),dtype=np.float32)
        dis_next = np.array(cv2.resize(dis_next, None, fx=0.5, fy=0.5, \
                              interpolation=cv2.INTER_AREA),dtype=np.float32)
    
    # calculate local averages    
    h, w = ref.shape
    mu_ref = np.zeros((h, w), dtype=np.float32)
    mu_dis = np.zeros((h, w), dtype=np.float32)
    
    scipy.ndimage.correlate1d(ref, window, 0, mu_ref, mode='reflect')
    scipy.ndimage.correlate1d(mu_ref, window, 1, mu_ref, mode='reflect')
    
    scipy.ndimage.correlate1d(dis, window, 0, mu_dis, mode='reflect')
    scipy.ndimage.correlate1d(mu_dis, window, 1, mu_dis, mode='reflect')
    
    # estimate local variances and conditional entropies in the spatial
    # domain for ith reference and distorted frames
    ss_ref, q_ref = est_params(ref - mu_ref, blk, sigma_nsq)
    spatial_ref = q_ref*np.log2(1+ss_ref)
    ss_dis, q_dis = est_params(dis - mu_dis, blk, sigma_nsq)
    spatial_dis = q_dis*np.log2(1+ss_dis)
    
    speed_s = np.nanmean(np.abs(spatial_ref.ravel() - spatial_dis.ravel()))
    speed_s_sn = np.abs(np.nanmean(spatial_ref.ravel() - spatial_dis.ravel()))
    
    ## frame differencing
    ref_diff = ref_next - ref;
    dis_diff = dis_next - dis;
    
    ## calculate local averages of frame differences
    mu_ref_diff = np.zeros((h, w), dtype=np.float32)
    mu_dis_diff = np.zeros((h, w), dtype=np.float32)
    
    scipy.ndimage.correlate1d(ref_diff, window, 0, mu_ref_diff, mode='reflect')
    scipy.ndimage.correlate1d(mu_ref_diff, window, 1, mu_ref_diff, mode='reflect')
    
    scipy.ndimage.correlate1d(dis_diff, window, 0, mu_dis_diff, mode='reflect')
    scipy.ndimage.correlate1d(mu_dis_diff, window, 1, mu_dis_diff, mode='reflect')
    
    """ Temporal SpEED
     estimate local variances and conditional entropies in the spatial
     domain for the reference and distorted frame differences """
     
    ss_ref_diff, q_ref = est_params(ref_diff - mu_ref_diff, blk, sigma_nsq)
    temporal_ref = q_ref*np.log2(1+ss_ref)*np.log2(1+ss_ref_diff)
    ss_dis_diff, q_dis = est_params(dis_diff - mu_dis_diff, blk, sigma_nsq)
    temporal_dis = q_dis*np.log2(1+ss_dis)*np.log2(1 + ss_dis_diff)
    
    speed_t = np.nanmean(np.abs(temporal_ref.ravel() - temporal_dis.ravel()));
    speed_t_sn = np.abs(np.nanmean(temporal_ref.ravel() - temporal_dis.ravel()));
    
    return speed_s, speed_s_sn, speed_t, speed_t_sn

def est_params(y, blk, sigma):
    """ 'ss' and 'ent' refer to the local variance parameter and the
        entropy at different locations of the subband
        y is a subband of the decomposition, 'blk' is the block size, 'sigma' is
        the neural noise variance """
    
    sizeim = np.floor(np.array(y.shape)/blk) * blk
    sizeim = sizeim.astype(int)
    y = y[:sizeim[0],:sizeim[1]].T
    
    temp = skimage.util.view_as_windows(np.ascontiguousarray(y), (blk,blk))\
    .reshape(-1,blk*blk).T
    
    cu = np.cov(temp, bias=1).astype(np.float32)
    
    eigval, eigvec = np.linalg.eig(cu)
    Q = np.matrix(eigvec)
    #L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
    L = np.matrix(np.diag(np.maximum(eigval, 0)))
    
    cu = Q*L*Q.T
    temp = skimage.util.view_as_blocks(np.ascontiguousarray(y), (blk,blk))\
    .reshape(-1,blk*blk).T
    
    L,Q = np.linalg.eigh(cu.astype(np.float64))
    L = L.astype(np.float32)
    #Estimate local variance parameters
    if np.max(L) > 0:
        ss = scipy.linalg.solve(cu, temp)
        ss = np.sum(ss*temp, axis=0)/(blk*blk)
        ss = ss.reshape((int(sizeim[1]/blk), int(sizeim[0]/blk))).T
    else:
        ss = np.zeros((sizeim/blk).astype(int),dtype=np.float32)
    
    L = L[L>0]
    
    #Compute entropy
    ent = np.zeros_like(ss, dtype=np.float32)
    for u in range(len(L)):
        ent += np.log2(ss*L[u]+sigma) + np.log(2*math.pi*np.exp(1));
        
    return ss, ent

def fread(fid, nelements, dtype):
     if dtype is str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array

def y4mFileRead(filePath,width, height,startFrame):
#    """Cut the YUV file at startFrame position for numFrame frames"""
    oneFrameNumBytes = int(width*height*1.5)
    with open(filePath, 'r+b') as file1:

        # header info
        line1 = file1.readline()

        # string of FRAME 
        line2 = file1.readline()

        frameByteOffset = len(line1)+(len(line2)+oneFrameNumBytes) * startFrame

        # each frame begins with the 5 bytes 'FRAME' followed by some zero or more characters"
        bytesToRead = oneFrameNumBytes + len(line2)

        file1.seek(frameByteOffset)
        y1 = fread(file1,height*width,np.uint8)
        y = np.reshape(y1,(height,width))
        return np.expand_dims(y,2)

distorted_yuv= glob.glob(os.path.join('/data/PV_VQA_Study/all_cut_upscaled_y4m_vids','*'))
def fps_from_content(content,fr):
    if(content=='EPLDay' or content=='EPLNight' or content=='Cricket1' or content=='Cricket2' or content=='USOpen'):
        if(fr=='HFR'):
            fps = 50
        else:
            fps = 25
    elif(content=='TNFF' or content=='TNFNFL'):
        if(fr=='HFR'):
            fps = 59.94
        else:
            fps = 29.97
    return str(fps)

def single_vid_speed(i):
    dis_vid = distorted_yuv[i]
    content = os.path.basename(dis_vid).split('_')[0]
    fps = fps_from_content(content,'HFR')
    begin_time = dis_vid.split('_')[-1]
    dis_FR = os.path.basename(dis_vid).split('_')[2]
    dis_fps = fps_from_content(content,dis_FR)
    print(dis_fps,fps)
    if(dis_fps==fps):
        ref_video = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_y4m_vids',content+'_SRC_SRC_SRC_SRC_'+begin_time[:-3]+'y4m')
    else:
        ref_video = os.path.join('/home/ubuntu/GREED/lbvfr/pseudo_reference_lbvfr/',content+'_SRC_SRC_SRC_SRC_'+begin_time[:-4]+'_pseudo_reference.y4m')

    width,height=int(3840),int(2160)
    speed_outname = os.path.join('./speed_features_PR/',os.path.splitext(os.path.basename(dis_vid))[0]+'.z')
    if(os.path.exists(speed_outname)):
        return
    print(ref_video,dis_vid,height,width,dis_fps,speed_outname)
    speed_list= []
    avg_window = gen_gauss_window(3, 7.0/6.0)

    frame_num= 0 
    while(True):
        try:
            ref_y = y4mFileRead(ref_video,width,height,frame_num)
            ref_y_next = y4mFileRead(ref_video,width,height,frame_num+1) 
            dis_y = y4mFileRead(dis_vid,width,height,frame_num)
            dis_y_next = y4mFileRead(dis_vid,width,height,frame_num+1)
        except Exception as e:
            print(e)
            print(frame_num, ' frames read')
            dump(speed_list,speed_outname)
            break
        speed = compute_speed(ref_y,ref_y_next,dis_y,dis_y_next,avg_window)
        speed_list.append(speed)
        frame_num = frame_num+1



    #speed_command = ['./run_speed.sh',ref_video,dis_vid,speed_outname,dis_fps]
    #try:
    #subprocess.check_call(speed_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return
def speed_refall_wrapper(ind,files):
    ref_f = files[ind]
    content = os.path.basename(ref_f).split('_')[1]
    print(content)
    dis_filenames = glob.glob("../../../hdr_yuv/4k*"+content)
    print(dis_filenames)
    Parallel(n_jobs=-1,verbose=1)(delayed(speed_video_wrapper)(ref_f,dis_f) for dis_f in dis_filenames)

def speed_video_wrapper(ref_f,dis_f):
    basename = os.path.basename(dis_f)
    dis_index = csv_df.index[csv_df['yuv'] == basename].tolist()[0]
    h =hs[dis_index]
    w = ws[dis_index]
    framenos = framenos_list[dis_index]
    speed_image_wrapper(ref_f,dis_f,framenos,h,w)

def speed_image_wrapper(ref_f,dis_f,framenos,h,w,adaptation='bilateral',use_adaptive_csf=True,use_non_overlapping_blocks=True,use_views=False):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    randlist = np.arange(framenos) # np.random.randint(0,framenos,10)

    score_df = pd.DataFrame([])
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    output_csv = os.path.join('./features/speed',dis_name+'.csv')
    if(os.path.exists(output_csv)==True):
        return
    with open(output_csv,'a') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['framenum','speed','nums','denoms'])
        for framenum in range(framenos):
            ref_y_pq,_,_ = hdr_yuv_read(ref_file_object,framenum,h,w)
            ref_y_pq_next,_,_ = hdr_yuv_read(ref_file_object,framenum+1,h,w)

            dis_y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum+1,h,w)
            dis_y_pq_next,_,_ = hdr_yuv_read(dis_file_object,framenum+1,h,w)

            if(use_adaptive_csf==True):
                # apply CSF here
                if(use_non_overlapping_blocks==True): # apply CSF on non-overlapping blocks of the image
                    csf_filtered_ref_y_pq = blockwise_csf(ref_y_pq,adaptation=adaptation)
                    csf_filtered_ref_y_pq_next = blockwise_csf(ref_y_pq,adaptation=adaptation)
                    csf_filtered_dis_y_pq = blockwise_csf(dis_y_pq,adaptation=adaptation)
                    csf_filtered_dis_y_pq_next = blockwise_csf(dis_y_pq,adaptation=adaptation)
                else: # sliding window; returns filtered value at center of each sliding window
                    csf_filtered_ref_y_pq = windows_csf(ref_y_pq,use_views=use_views)
                    csf_filtered_ref_y_pq_next = windows_csf(ref_y_pq,use_views=use_views)
                    csf_filtered_dis_y_pq = windows_csf(dis_y_pq,use_views=use_views)
                    csf_filtered_dis_y_pq_next = windows_csf(dis_y_pq,use_views=use_views)

                # standard VIF but without CSF
                speed_val = compute_speed(csf_filtered_ref_y_pq,csf_filtered_ref_y_pq_next,csf_filtered_dis_y_pq,csf_filtered_dis_y_pq_next,avg_window)
            else:
                # standard VIF 
                speed_val = compute_speed(ref_y_pq,ref_y_pq_next,dis_y_pq,dis_y_pq_next,avg_window)

Parallel(n_jobs=5)(delayed(single_vid_speed)(i) for i in range(len(distorted_yuv)))
