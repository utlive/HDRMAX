input=$1
output=$2
csvpth=$3
njobs=$4
range=$5
python hdrvmaf_features.py $input $output $csvpth --space ycbcr  --nonlinear local_exp --parameter 0.5 --channel 0  --vif --dlm --njobs $njobs --frame_range $range
python hdrvmaf_features.py $input $output $csvpth --space ycbcr  --nonlinear local_exp --parameter 5 --channel 0  --vif --dlm --njobs $njobs --frame_range $range
python hdrvmaf_features.py $input $output $csvpth --space ycbcr  --nonlinear none --parameter 2 --channel 0  --vif --dlm --njobs $njobs --frame_range $range
python hdrvmaf_features.py $input $output $csvpth --space ycbcr  --nonlinear local_exp --parameter 1 --channel 0  --vif --dlm --njobs $njobs --frame_range $range
