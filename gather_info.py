import os
import glob
import pandas as pd

video_list = glob.glob('/home/ubuntu/data2/VQA/2022_summer/test_videos_yuv/*.yuv')
info_df = pd.DataFrame()
for vid in video_list:
    base_name = os.path.basename(vid)   
    content = base_name.split('_')[0]
    if content == 'anchor':
        continue
    else:
        fps = base_name.split('_')[6]
        number_frames = int(os.path.getsize(vid) / (2160*3840*2*1.5))
        print(base_name, fps, number_frames)
        mode = base_name.split('_')[2]
        if base_name.find('football') != -1 and mode == 'CBR':
            reference = f'{content}_1080p_CBR_30000_na_na_50_on_ST_.yuv'
        elif base_name.find('football') != -1 and mode == 'QVBR':
            reference = f'{content}_1080p_QVBR_na_9.0_30000_50_off_na_.yuv'
        elif base_name.find('football') == -1 and mode == 'CBR':
            reference = f'{content}_2160p_CBR_60000_na_na_50_on_ST_.yuv'
        elif base_name.find('football') == -1 and mode == 'QVBR':
            reference = f'{content}_2160p_QVBR_na_9.0_60000_50_off_na_.yuv'
        else:
            raise
    # put everthing in a dataframe
    info_dict = {'video': base_name,
                'fps': fps,  
                'number_frames': number_frames,
                'reference': reference} 
    df = pd.DataFrame(info_dict, index=[0])
    # append to the dataframe

    info_df = pd.concat([info_df, df])

        
info_df.to_csv('info_df.csv')