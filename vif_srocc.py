import numpy as np
import pandas as pd
import os
import pandas as pd
from joblib import load,dump
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

filenames = glob.glob('./features/vif/*.csv')
all_vif = []
all_vvif = []
all_mos = []

score_df = pd.read_csv('/home/josh-admin/hdr/fall21_score_analysis/fall21_data.csv')

def results(all_preds,all_mos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_mos = np.asarray(all_mos)
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          all_preds, all_mos, p0=0.5*np.ones((5,)), maxfev=20000)

    preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    preds_srocc = spearmanr(preds_fitted,all_mos)
    preds_lcc = pearsonr(preds_fitted,all_mos)
    preds_rmse = np.sqrt(np.mean(preds_fitted-all_mos)**2)
    print('SROCC:')
    print(preds_srocc[0])
    print('LCC:')
    print(preds_lcc[0])
    print('RMSE:')
    print(preds_rmse)
    print(len(all_preds),' videos were read')

upscaled_names = [v+'_upscaled' for v in score_df["video"]]
for f in filenames:
    vid_name= os.path.splitext(os.path.basename(f))[0]

    print(vid_name)

    logit_vif_df = pd.read_csv(os.path.join('./features/vif_logit_global_delta3',vid_name+'.csv'))
    print(len(logit_vif_df))
    print(logit_vif_df)
    if(len(logit_vif_df)>10):
        logit_vif_df = logit_vif_df.iloc[11:,:]

    vif_df = pd.read_csv(f) 
    print('vif below')
    print(vif_df)
    print('logit vif below')
    print(logit_vif_df)
    vif = np.mean(vif_df["vif"])
    print(logit_vif_df["vif"])
    logit_vif = np.mean(logit_vif_df["vif"])
    combined_vif = vif+logit_vif

    print(vid_name,combined_vif)
    # note this is combined VIF 
    all_vif.append(vif)
    index = upscaled_names.index(vid_name)
    mos = score_df['dark_mos'].iloc[index]
    all_mos.append(mos)


results(all_vif,all_mos)

