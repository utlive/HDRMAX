## Change Log
### 2024-03-19
- Changed feature extraction, main, and prediction code. The current version calculates the score using the HDRMAX single nonlienar. The model files are also retrained. 

### 2024-03-05
- Changed the code to remove extra computation. The program now should be much faster. 


### 2023-04-20
- Added the HDR SSIM and HDR MS-SSIM
- Added main module.

### 2023-03-17
- Added the ability to train the model on custom datasets. 
- Added the HDR AQ model. The program to obtain the VIF and DLM features after nonlinear transform. This implementation assumes that the input video is in YUV420p10le format, and are 3840x2160 pixels.

## HDR-VMAF
This is the code implementation of the HDR-VMAF, HDR MS-SSIM and HDR-SSIM mentioned in the paper "Making Video Quality Assessment Models Robust to Bit Depth". This implementation has three models, the HDR-VMAF, HDR MS-SSIM and HDR-SSIM model. All of them are full reference HDR VQA models. 

### Requirements
---

**Python Version Requirement:**
This project requires Python version 3.9-3.10. This is due to certain dependencies in the `requirements.txt` file which are not compatible with later versions of Python.

**Installing Dependencies:**
To install the required packages, use the following command:
```
pip install -r requirements.txt
```
A frozen package list is also included `requirements_frozen.txt`, which has the tested versions of the packages, working with python3.9.

**Troubleshooting OpenCV Installation:**
Some users may encounter an error when importing OpenCV, typically due to missing OpenGL support. This issue can be resolved by installing the `libgl1-mesa-glx` package. Run the following commands:
```
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

---
### Usage

It is recommended to use the `main.py`. 

```
usage: main.py [-h] --input INPUT --output OUTPUT --csvpth CSVPTH [--njobs NJOBS] --frame_range FRAME_RANGE {hdrvmaf,ssim-hdrmax,msssim-hdrmax}

positional arguments:
  {hdrvmaf,ssim-hdrmax,msssim-hdrmax}
                        Select processing mode

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input file path
  --output OUTPUT       Output file path
  --csvpth CSVPTH       CSV path
  --njobs NJOBS         Number of jobs
  --frame_range FRAME_RANGE
                        Frame range
  --mode 
```

Here `INPUT` is the path to the source videos, the yuv files, they need to have the resolutions 3840*2160. `OUTPUT` can be an empty folder for the output features, `CSVPTH` is the csv file that includes the yuv file names of the input files, fps, and the original width and hight of the video, and the start frame index and end frame index if the `FRAME_RANGE` parameter is set to `file`. See info.csv for an example. `njob` is the number of jobs runing.
Note: Please have a clearn `out_root` folder each time before running the script.

If you wish to obtain features only or wish to use other models, please follow the instructions below:


`hdrvmaf.py` extracts hdrvmaf features. Here is it's usage:

```
        usage: hdrvmaf.py [-h] [--space SPACE] [--nonlinear NONLINEAR]
                        [--parameter PARAMETER] [--channel CHANNEL] [--vif] [--dlm]

        optional arguments:
        -h, --help            show this help message and exit
        --space SPACE         choose which color space. Support 'ycbcr' and 'lab'.
        --nonlinear NONLINEAR
                                select the nonliearity. Support
                                'global_logit','local_logit',
                                'local_m_exp','global_m_exp',
                                'global_exp' or 'none'.
        --parameter PARAMETER
                                the parameter for the nonliear. Use with --nonliear
        --channel CHANNEL     indicate which channel to process. Please provide 0,
                                1, or 2
        --vif                 obtaining vif output.
        --dlm                 obtaining vif output.
        --frame_range FRAME_RANGE
                              'all' or 'file'. if 'all', the whole video is used to estimate
                                the quality. if 'file', the video uses the 'start' and 'end'
                                columns in the csv file to estimate the quality.
```

The current version also takes a few hardcoded variables: `csv_file_vidinfo` is the csv file that includes the yuv file names of the input files, number of frames, fps, and the original width and hight of the video. `vid_pth` is the path to the source videos, the yuv files. Note the yuv files are upscaled to 3840x2160 (pix_fmt:yuv420p10le).  `out_root_vif` is the output path of the vif features. `out_root_dlm` is the output path of the dlm features.

A simpler way is to utilize the bash script:
```bash extractfeatures.sh vid_pth out_root csv_file_vidinfo njob frame_range```
Here `vid_pth` is the path to the source videos, the yuv files, they need to have the resolutions 3840*2160. `out_root` can be an empty folder for the output features, `csv_file_vidinfo` is the csv file that includes the yuv file names of the input files, fps, and the original width and hight of the video, and the start frame index and end frame index if the `frame_range` parameter is set to `file`. See info.csv for an example. `njob` is the number of jobs runing.
Note: Please have a clearn `out_root` folder each time before running the script.

For example:
```bash extractfeatures.sh  /mnt/6b81aa53-cdfa-48cc-8a07-49e874756570/video/live_hdrsdr/yuv/yuv ./tmp/feats   info.csv  1 all```

## Train on a Custom Dataset

To train the model on a custom dataset, follow these steps:

1. Extract the features from your custom dataset as described in the previous section.
2. Run the `train_model.py` script with the appropriate arguments.

The script accepts the following arguments:

- `feature_path`: Path to the folder containing the features extracted from your custom dataset.
- `score_csv`: Path to the score file. This should be a CSV file with the following columns: video, score, content. It is critical that the video column has the same name as the video files in the feature folder.
- `--scaler_name` (optional): The name of the Scaler. Defaults to 'model_scaler.pkl'.
- `--svr_name` (optional): The name of the SVR. Defaults to 'model_svr.pkl'.



## Predict Quality Scores for Videos

The `predict.py` script uses the extracted features and trained models to predict quality scores for each video. To run the script, you'll need to provide the following arguments:

- `feature_path`: Path to the folder containing the features extracted from the videos.
- `output_name`: Name of the output file where the predictions will be saved, defaults to 'predict.csv'.
- `--model` (optional): Which model to use. Options: LIVEHDR, LIVEAQ, CUSTOM. Defaults to 'LIVEHDR'. The HDRAQ model is trained on the UTA spring dataset. The HDRLIVE model is trained on the UTA 2021 fall dataset.
- `--svr_name` (optional): SVR name, to be used with the CUSTOM model.
- `--scaler_name` (optional): Scaler name, to be used with the CUSTOM model.



Please contact Zaixi Shang (zxshang@utexas.edu) if you have any questions.
