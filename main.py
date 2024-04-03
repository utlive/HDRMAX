import argparse
import os


def main(args):
    input = args.input
    output = args.output
    csvpth = args.csvpth
    njobs = args.njobs
    frame_range = args.frame_range
    # check if the path is empty
    if os.path.exists(output) and os.path.isdir(output):
        if os.listdir(output):
            raise RuntimeError(f"The feature path: {output} is not empty. Please remove all the content in the directory or choose a different directory.")

    if args.mode == 'hdrvmaf':
        commands = [
            f"python hdrvmaf_features.py {input} {output} {csvpth} --space ycbcr --nonlinear local_m_exp  --channel 0 --vif --dlm --njobs {njobs} --frame_range {frame_range}",
            f"python hdrvmaf_features.py {input} {output} {csvpth} --space ycbcr --nonlinear none --parameter 2 --channel 0 --vif --dlm --njobs {njobs} --frame_range {frame_range}",
            f"python predict.py {output} out.csv"
        ]
    elif args.mode == 'ssim-hdrmax':
        commands = [
            f"python hdrvmaf_features.py {input} {output} {csvpth} --space ycbcr --nonlinear local_m_exp --channel 0 --vif --dlm --njobs {njobs} --frame_range {frame_range}",
            f"python extract_ssim.py {input} {output} {csvpth} --space ycbcr  --channel 0 --njobs {njobs} --frame_range {frame_range}",
            f"python predict_hdrssim.py {output} out.csv",
        ]
    elif args.mode == 'msssim-hdrmax':
        commands = [
            f"python hdrvmaf_features.py {input} {output} {csvpth} --space ycbcr --nonlinear local_m_exp --channel 0 --vif --dlm --njobs {njobs} --frame_range {frame_range}",
            f"python extract_msssim.py {input} {output} {csvpth} --space ycbcr  --channel 0 --njobs {njobs} --frame_range {frame_range}",
            f"python predict_hdrmsssim.py {output} out.csv",
        ]
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    for cmd in commands:
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['hdrvmaf', 'ssim-hdrmax', 'msssim-hdrmax'], help='Select processing mode')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--csvpth', required=True, help='CSV path')
    parser.add_argument('--njobs', type=int, default=1, help='Number of jobs')
    parser.add_argument('--frame_range', required=True, help='Frame range')

    args = parser.parse_args()
    main(args)
