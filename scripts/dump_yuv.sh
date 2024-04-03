#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <inputfile.mp4>"
    exit 1
fi

inputfile=$1
outputfile="${inputfile%.*}.yuv"

ffmpeg -i "$inputfile" -vf "scale=3840x2160" -pix_fmt yuv420p10le "$outputfile"

echo "Upscaling completed. Output file: $outputfile"
