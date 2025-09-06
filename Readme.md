## Change Log
### 2025-09-06
- Refactored code, moved feature extraction functions to feature_utils.py
- Added SSIM and MS-SSIM feature extraction functions
- Created unified prediction script predict_unified.py, supporting VMAF, SSIM, and MS-SSIM models

### 2025-08-27
- Fixed numerical stability issues in m_exp function
- Added comprehensive .gitignore file for Python projects
- Updated README with nonlinear transformation explanation and HDR-ChipQA paper reference
- Improved code documentation and comments

### 2024-03-19
- Changed feature extraction, main, and prediction code. The current version calculates the score using the HDRMAX single nonlienar (`local_m_exp`). The model files are also retrained.

### 2024-03-05
- Changed the code to remove extra computation. The program now should be much faster.


### 2023-04-20
- Added the HDR SSIM and HDR MS-SSIM
- Added main module.

### 2023-03-17
- Added the ability to train the model on custom datasets. 

## HDR-VMAF
This is the code implementation of the HDR-VMAF, HDR MS-SSIM and HDR-SSIM mentioned in the paper. This implementation has three models, the HDR-VMAF, HDR MS-SSIM and HDR-SSIM model. All of them are full reference HDR VQA models.

### Nonlinear Transformation
HDRMAX employs the local expansive nonlinearity technique from the HDR-ChipQA paper by Ebenezer et al.* This approach addresses the challenge that HDR-specific distortions at brightness/color extremes are often masked by features computed on standard dynamic range regions. The method applies an expansive nonlinearity `f(y) = exp(|y|×4) - 1` on locally normalized pixel values (17×17 windows mapped to [-1,1]), which expands extreme brightness ranges while compressing mid-ranges. This transformation enables better detection of distortions in very bright and dark regions that are critical for HDR video quality assessment.

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
usage: main.py [-h] {hdrvmaf,ssim-hdrmax,msssim-hdrmax} --input INPUT --output OUTPUT --csvpth CSVPTH [--njobs NJOBS] --frame_range FRAME_RANGE

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

To train the model on a custom dataset, you can use one of the three training scripts depending on which model you want to train:

### 1. Training VMAF Model

Use the `train_model.py` script to train the VMAF model:

```
python train_model.py feature_path score_csv [--scaler_name SCALER_NAME] [--svr_name SVR_NAME]
```

### 2. Training SSIM Model

Use the `train_modelssim.py` script to train the SSIM model:

```
python train_modelssim.py feature_path score_csv [--scaler_name SCALER_NAME] [--svr_name SVR_NAME]
```

### 3. Training MS-SSIM Model

Use the `train_modelmsssim.py` script to train the MS-SSIM model:

```
python train_modelmsssim.py feature_path score_csv [--scaler_name SCALER_NAME] [--svr_name SVR_NAME]
```

All three scripts accept the following arguments:

- `feature_path`: Path to the folder containing the features extracted from your custom dataset.
- `score_csv`: Path to the score file. This should be a CSV file with the following columns: video, score, content. It is critical that the video column has the same name as the video files in the feature folder.
- `--scaler_name` (optional): The name of the Scaler. Defaults to 'model_scaler.pkl'.
- `--svr_name` (optional): The name of the SVR. Defaults to 'model_svr.pkl'.

For example, to train the SSIM model:
```
python train_modelssim.py ./features-test-ssim score_file.csv --scaler_name models/scaler/ssim_scaler.pkl --svr_name models/svr/ssim_svr.pkl
```

## Predict Quality Scores for Videos

The `predict_unified.py` script uses the extracted features and trained models to predict quality scores for videos. This unified script supports all three model types: VMAF, SSIM, and MS-SSIM.

```
python predict_unified.py feature_path output_name --model MODEL
```

Arguments:
- `feature_path`: Path to the folder containing the features extracted from the videos.
- `output_name`: Name of the output file where the predictions will be saved.
- `--model`: Which model to use. Options: VMAF, SSIM, MSSSIM. Default is 'VMAF'.

Examples:

1. Using VMAF model:
```
python predict_unified.py ./features-test out_vmaf.csv --model VMAF
```

2. Using SSIM model:
```
python predict_unified.py ./features-test-ssim out_ssim.csv --model SSIM
```

3. Using MS-SSIM model:
```
python predict_unified.py ./features-test-msssim out_msssim.csv --model MSSSIM
```

Please contact Zaixi Shang (zxshang@utexas.edu) if you have any questions.

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------

Copyright (c) 2024 Laboratory for Image and Video Engineering (LIVE)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The following papers are to be cited in the bibliography whenever the software is used as:

-Z. Shang, J. P. Ebenezer, A. C. Bovik, Y. Wu, H. Wei, and S. Sethu- raman, "Subjective assessment of high dynamic range videos under different ambient conditions," in 2022 IEEE International Conference on Image Processing (ICIP), 2022

-J. P. Ebenezer, Z. Shang, Y. Wu, H. Wei, S. Sethuraman and A. C. Bovik, "Making Video Quality Assessment Models Robust to Bit Depth," in IEEE Signal Processing Letters, vol. 30, pp. 488-492, 2023, doi: 10.1109/LSP.2023.3268602.

-Z. Shang et al., "A Study of Subjective and Objective Quality Assessment of HDR Videos," in IEEE Transactions on Image Processing, vol. 33, pp. 42-57, 2024, doi: 10.1109/TIP.2023.3333217.

-*J. P. Ebenezer, Z. Shang, Y. Wu, H. Wei, S. Sethuraman and A. C. Bovik, "HDR-ChipQA: No-reference quality assessment on High Dynamic Range videos," Signal Processing: Image Communication, vol. 129, pp. 117191, 2024, doi: 10.1016/j.image.2024.117191.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------
