import numpy as np
from pywt import wavedec2,waverec2
from utils.hdr_utils import hdr_yuv_read

import matplotlib.pyplot as plt

def dlm_decouple(pyr_ref, pyr_dist):
    eps = 1e-30
    n_levels = len(pyr_ref)
    pyr_rest = []
    pyr_add = []

    for level in range(n_levels):
        # for contrast enhancement
        psi_ref = np.arctan(pyr_ref[level][1] / (pyr_ref[level][0] + eps)) + np.pi*(pyr_ref[level][0] <= 0)
        psi_dist = np.arctan(pyr_dist[level][1] / (pyr_dist[level][0] + eps)) + np.pi*(pyr_dist[level][0] <= 0)

        psi_diff = 180*np.abs(psi_ref - psi_dist)/np.pi
        mask = (psi_diff < 1)
        level_rest = []
        for i in range(3):
            k = np.clip(pyr_dist[level][i] / (pyr_ref[level][i] + eps), 0.0, 1.0)
            level_rest.append(k * pyr_ref[level][i])
            level_rest[i][mask] = pyr_dist[level][i][mask]

        pyr_rest.append(tuple(level_rest))

    for level_dist, level_rest in zip(pyr_dist, pyr_rest):
        level_add = []
        for i in range(3):
            level_add.append(level_dist[i] - level_rest[i])
        pyr_add.append(tuple(level_add))

    return pyr_rest, pyr_add

def dlm(img_ref, img_dist, wavelet='db2', border_size=0.2, full=True, csf='adm'):
    n_levels = 4

    pyr_ref = wavedec2(img_ref, wavelet, 'reflect', n_levels)
    pyr_dist = wavedec2(img_dist, wavelet, 'reflect', n_levels)

    print(len(pyr_ref))
    approx_ref,approx_dist = pyr_ref[0].copy(),pyr_dist[0].copy()
    print(approx_ref.shape)
#    blocks_list = view_as_blocks(y_linear[:max_h,:max_w].copy(),(h_win,w_win))
#    blocks =blocks_list.reshape(-1,blocks_list.shape[2],blocks_list.shape[3])

    # Ignore approximation coefficients
    del pyr_ref[0], pyr_dist[0]
    pyr_ref.reverse()
    pyr_dist.reverse()

    pyr_rest, pyr_add = dlm_decouple(pyr_ref, pyr_dist)
    pyr_rest.reverse()
    pyr_rest.insert(0,approx_dist)
    rest = waverec2(pyr_rest,wavelet, 'reflect')
    return rest

def dlm_image_wrapper(ref_f,dis_f,framenum,h,w):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)

    ref_y_pq,_,_ = hdr_yuv_read(ref_file_object,framenum,h,w)

    dis_y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)
    print(np.amax(dis_y_pq),np.amin(dis_y_pq))
    dis_y_pq = dis_y_pq+(np.random.rand(h,w)-0.5)*30
    print(np.amax(dis_y_pq),np.amin(dis_y_pq))
    # standard DLM
    rest = dlm(ref_y_pq,dis_y_pq)
    add = dis_y_pq-rest
    return rest,add,dis_y_pq

ref_f = '../../data/reference_Blanc1.yuv'
dis_f = '../../data/4k_cbr3Mbps_Blanc1.yuv'
framenos = 1
h,w = 2160,3840
rest,add,dis_y_pq = dlm_image_wrapper(ref_f,dis_f,framenos,h,w)

plt.figure()
plt.imshow(add,'gray')
plt.savefig('additive.png')


plt.figure()
plt.imshow(rest,'gray')
plt.savefig('restored.png')


plt.figure()
plt.imshow(dis_y_pq,'gray')
plt.savefig('distorted.png')
