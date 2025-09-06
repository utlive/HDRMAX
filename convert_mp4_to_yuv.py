#!/usr/bin/env python3
"""
Multi-threaded MP4 to YUV converter using FFmpeg
Converts MP4 files to YUV format with 4K upscaling and yuv420p10le pixel format
"""

import os
import subprocess
import threading
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from pathlib import Path

def convert_mp4_to_yuv(input_file, output_dir, max_workers=4):
    """
    Convert a single MP4 file to YUV format with 4K upscaling
    
    Args:
        input_file (str): Path to input MP4 file
        output_dir (str): Directory to save YUV output
        max_workers (int): Maximum number of worker threads
    """
    input_path = Path(input_file)
    output_filename = input_path.stem + "_4k.yuv"
    output_path = Path(output_dir) / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command for upscaling to 4K and converting to yuv420p10le
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(input_file),
        '-vf', 'scale=3840:2160:flags=lanczos',  # Upscale to 4K (3840x2160) using Lanczos
        '-pix_fmt', 'yuv420p10le',  # Output pixel format
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    try:
        print(f"Starting conversion: {input_path.name} -> {output_filename}")
        
        # Run FFmpeg command
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Successfully converted: {input_path.name}")
        return True, str(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {input_path.name}: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False, str(output_path)
    except Exception as e:
        print(f"✗ Unexpected error converting {input_path.name}: {e}")
        return False, str(output_path)

def process_videos_multithreaded(input_dir, output_dir, max_workers=4):
    """
    Process multiple MP4 files using multithreading
    
    Args:
        input_dir (str): Directory containing MP4 files
        output_dir (str): Directory to save YUV outputs
        max_workers (int): Maximum number of worker threads
    """
    # Find all MP4 files in input directory
    mp4_pattern = os.path.join(input_dir, "**/*.mp4")
    mp4_files = glob.glob(mp4_pattern, recursive=True)
    
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files to convert")
    print(f"Using {max_workers} worker threads")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        future_to_file = {
            executor.submit(convert_mp4_to_yuv, mp4_file, output_dir): mp4_file 
            for mp4_file in mp4_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            mp4_file = future_to_file[future]
            try:
                success, output_path = future.result()
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            except Exception as e:
                print(f"✗ Exception occurred for {mp4_file}: {e}")
                failed_conversions += 1
    
    print("-" * 50)
    print(f"Conversion completed!")
    print(f"✓ Successful: {successful_conversions}")
    print(f"✗ Failed: {failed_conversions}")
    print(f"Total: {len(mp4_files)}")

def main():
    parser = argparse.ArgumentParser(description="Convert MP4 files to YUV with 4K upscaling")
    parser.add_argument(
        "--input-dir", 
        default="/mnt/data/videos/LIVE_HDR/videos/videos_aligned",
        help="Input directory containing MP4 files (default: /mnt/data/videos/LIVE_HDR/videos/videos_aligned/)"
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/videos/LIVE_HDR/videos/yuv_upscaled/",
        help="Output directory for YUV files (default: /mnt/data/videos/LIVE_HDR/videos/yuv_upscaled/)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not available in PATH")
        return 1
    
    print("MP4 to YUV Converter with 4K Upscaling")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Worker threads: {args.threads}")
    print(f"Target resolution: 3840x2160 (4K)")
    print(f"Pixel format: yuv420p10le")
    print("=" * 50)
    
    # Start processing
    process_videos_multithreaded(args.input_dir, args.output_dir, args.threads)
    
    return 0

if __name__ == "__main__":
    exit(main())
