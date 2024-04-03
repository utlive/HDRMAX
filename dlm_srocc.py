import numpy as np
import pandas as pd
import os
import pandas as pd
from joblib import load,dump
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

filenames = glob.glob('./features/dlm_logit_global_delta3/*.csv')
all_dlm = []
all_vdlm = []
all_dmos = []

score_df = pd.read_csv('/home/josh-admin/hdr/fall21_score_analysis/fall21_data.csv')

def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)

    preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean(preds_fitted-all_dmos)**2)
    print('SROCC:')
    print(preds_srocc[0])
    print('LCC:')
    print(preds_lcc[0])
    print('RMSE:')
    print(preds_rmse)
    print(len(all_preds),' videos were read')

upscaled_names = [v+'_upscaled' for v in score_df["video"]]
for f in filenames:
    if('reference' in f):
        continue
    vid_name= os.path.splitext(os.path.basename(f))[0]
    print(vid_name)
    print(f)
    dlm_df = pd.read_csv(f) 
    dlm = np.mean(dlm_df["dlm"])
    print(vid_name,dlm)
    all_dlm.append(dlm)
    index = upscaled_names.index(vid_name)
    mos = score_df['dark_mos'].iloc[index]
    all_dmos.append(mos)


results(all_dlm,all_dmos)

