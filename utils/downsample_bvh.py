"""
BVH Motion Data Downsampling Utility

Provides functionality for downsampling BVH motion capture files to a target frame rate.
Uses pymo library for parsing and processing BVH files.

Usage:
    Single file: downsample_bvh(input_file, output_file, target_fps)
    Batch process: batch_process_bvh_files(input_dir, output_dir, target_fps)
"""

import os
import sys
import argparse
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import DownSampler
from pymo.writers import BVHWriter


def downsample_bvh(input_file, output_file, target_fps):
    """Downsample BVH file to target FPS and export.

    Args:
        input_file: Path to input BVH file.
        output_file: Path to output BVH file.
        target_fps: Target frame rate.

    Returns:
        True if successful.
    """
    parser = BVHParser()
    mocap_data = parser.parse(input_file)

    if target_fps > (1 / mocap_data.framerate):
        print(f"error, target_fps = {target_fps} must < original fps = {1 / mocap_data.framerate}")

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=target_fps, keep_all=False)),
    ])
    feature_data = data_pipe.transform([mocap_data])

    writer = BVHWriter()
    with open(output_file, 'w') as f:
        writer.write(feature_data[0], f)

    return True


def batch_process_bvh_files(input_dir, output_dir, target_fps):
    """Batch process all BVH files in a directory.

    Args:
        input_dir: Input directory path containing BVH files.
        output_dir: Output directory path for downsampled files.
        target_fps: Target frame rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    bvh_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.bvh')]

    if not bvh_files:
        print(f"No BVH files found in {input_dir}")
        return

    print(f"Found {len(bvh_files)} BVH files, processing...")

    success_count = 0
    for bvh_file in tqdm(bvh_files):
        input_path = os.path.join(input_dir, bvh_file)
        output_path = os.path.join(output_dir, bvh_file)

        try:
            result = downsample_bvh(input_path, output_path, target_fps)
            if result:
                success_count += 1
            else:
                print(f"Failed to process {bvh_file}: frame rate issue or incompatible format")
        except Exception as e:
            print(f"Error processing {bvh_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"Processing complete! Successfully processed {success_count}/{len(bvh_files)} files")
    print(f"Downsampled files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch downsample BVH files")
    parser.add_argument("--input_dir", "-i", required=True, help="Input BVH directory path")
    parser.add_argument("--output_dir", "-o", required=True, help="Output BVH directory path")
    parser.add_argument("--target_fps", "-t_fps", type=int, default=30, help="Target frame rate")

    args = parser.parse_args()

    batch_process_bvh_files(args.input_dir, args.output_dir, args.target_fps)
