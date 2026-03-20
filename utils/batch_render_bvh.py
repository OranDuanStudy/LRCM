"""
Batch BVH to Video Rendering Utility

Traverses all subdirectories to find BVH files and renders them to videos.
Supports adding audio (from .wav files) and text overlay (from gen_texts.json).

Usage:
    python utils/batch_render_bvh.py --input_dir <path> [options]

Example:
    python utils/batch_render_bvh.py --input_dir ./bvh_files --output_dir ./videos \\
        --gen_texts_json ./gen_texts.json --view third
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4, render_mp4_second_person


def find_gen_texts_json(input_dir):
    """
    Find gen_texts.json file in the input directory tree.

    Args:
        input_dir: Root directory to search for gen_texts.json

    Returns:
        Path to gen_texts.json file, or None if not found
    """
    input_path = Path(input_dir)

    # First check if gen_texts.json exists in the root directory
    root_json = input_path / "gen_texts.json"
    if root_json.exists():
        return root_json

    # Search recursively
    for json_file in input_path.rglob("gen_texts.json"):
        return json_file

    return None


def load_gen_texts(json_path):
    """
    Load gen_texts.json file.

    Args:
        json_path: Path to gen_texts.json file

    Returns:
        Dictionary with file names as keys and text data as values
    """
    if json_path is None or not json_path.exists():
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_text_for_file(bvh_filename, gen_texts):
    """
    Get text data for a BVH file from gen_texts dictionary.

    Args:
        bvh_filename: BVH file name (with or without extension)
        gen_texts: Dictionary from gen_texts.json

    Returns:
        Tuple of (global_text, local_text) or (None, None) if not found
    """
    # Remove .bvh extension if present
    base_name = bvh_filename.replace('.bvh', '')

    # Try to find matching key in gen_texts
    if base_name in gen_texts:
        data = gen_texts[base_name]
        global_text = data.get('global', '')
        local_text = data.get('local', '')
        return global_text, local_text

    # Try with _00 suffix
    if f"{base_name}_00" in gen_texts:
        data = gen_texts[f"{base_name}_00"]
        global_text = data.get('global', '')
        local_text = data.get('local', '')
        return global_text, local_text

    return None, None


def find_audio_file(bvh_file):
    """
    Find the audio file (.wav) corresponding to a BVH file.

    Args:
        bvh_file: Path to the BVH file

    Returns:
        Path to the audio file, or None if not found
    """
    bvh_path = Path(bvh_file)
    bvh_dir = bvh_path.parent
    bvh_name = bvh_path.stem

    # Look for .wav file with the same name
    wav_file = bvh_dir / f"{bvh_name}.wav"
    if wav_file.exists():
        return wav_file

    return None


def add_audio_with_ffmpeg(video_path, audio_path, output_path):
    """
    Add audio to video using ffmpeg command line.

    Args:
        video_path: Path to the input video file
        audio_path: Path to the audio file
        output_path: Path to the output video file

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        # Use ffmpeg to merge video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"  FFmpeg error: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  FFmpeg not found. Please install ffmpeg: apt-get install ffmpeg")
        return False
    except Exception as e:
        print(f"  Error adding audio with ffmpeg: {e}")
        return False


def add_audio_and_text_to_video(video_path, audio_path, global_text, local_text,
                                 output_path, font_size=48, bold=True):
    """
    Add audio and high-contrast text overlay to a video using ffmpeg.

    Args:
        video_path: Path to the input video file
        audio_path: Path to the audio file (.wav)
        global_text: Global style text (top-left corner)
        local_text: Local action text (top-right corner)
        output_path: Path to the output video file
        font_size: Font size for text overlay
        bold: Whether to use bold font

    Returns:
        True if successful, False otherwise
    """
    import shutil
    import tempfile

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Check if we need text overlay
        needs_text = global_text or local_text

        if not needs_text and not audio_path:
            # No text or audio needed
            if video_path != output_path:
                shutil.copy2(video_path, output_path)
            return True

        # Use a temp directory for intermediate files
        temp_dir = tempfile.mkdtemp()

        # Build ffmpeg command
        import subprocess

        # Start with input video
        cmd = ['ffmpeg', '-y', '-i', str(video_path)]

        # Add audio input if needed
        if audio_path and os.path.exists(audio_path):
            cmd.extend(['-i', str(audio_path)])

        # Build complex filter for text overlay
        filters = []
        if global_text:
            # Escape special characters in text
            safe_text = global_text.replace("'", "\\'").replace(":", "\\:")
            filters.append(f"drawtext=text='{safe_text}':fontsize={font_size}:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=2")

        if local_text:
            # For right-aligned text, we need to calculate x position
            # This will be done after we get video width
            safe_text = local_text.replace("'", "\\'").replace(":", "\\:")
            filters.append(f"drawtext=text='{safe_text}':fontsize={font_size}:fontcolor=yellow:x=w-tw-20:y=20:box=1:boxcolor=black@0.5:boxborderw=2")

        # Combine filters
        if filters:
            filter_str = ','.join(filters)
            cmd.extend(['-vf', filter_str])

        # Add output options
        if audio_path and os.path.exists(audio_path):
            cmd.extend(['-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-c:a', 'aac', '-shortest'])
        else:
            cmd.extend(['-c:v', 'libx264'])

        cmd.append(str(output_path))

        # Debug: print the ffmpeg command
        print(f"  FFmpeg command: {' '.join(cmd)}")

        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        if result.returncode == 0:
            return True
        else:
            print(f"  FFmpeg error: {result.stderr}")
            # Fallback: just copy the video
            shutil.copy2(video_path, output_path)
            return True

    except Exception as e:
        print(f"  Error adding audio and text: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp directory on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        # Fallback: just copy the video
        shutil.copy2(video_path, output_path)
        return True


def find_bvh_files(input_dir):
    """
    Recursively find all BVH files in the input directory and its subdirectories.

    Args:
        input_dir: Root directory to search for BVH files

    Returns:
        List of tuples: (bvh_file_path, relative_path_from_input_dir)
    """
    bvh_files = []
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for bvh_file in input_path.rglob("*.bvh"):
        relative_path = bvh_file.relative_to(input_path)
        bvh_files.append((str(bvh_file), str(relative_path)))

    return bvh_files


def render_single_bvh(bvh_file, output_file, gen_texts=None, view="third", axis_scale=200,
                       elev=45, azim=45, track_character=True, add_audio=True,
                       font_size=48):
    """
    Render a single BVH file to video with optional audio and text overlay.

    Args:
        bvh_file: Path to the BVH file
        output_file: Path to the output video file (without extension)
        gen_texts: Dictionary containing text data for each file
        view: Camera view - "third" for fixed third-person, "second" for second-person tracking
        axis_scale: Axis scale for the view
        elev: Elevation angle for third-person view
        azim: Azimuth angle for third-person view
        track_character: Whether to track character movement (third-person view only)
        add_audio: Whether to add audio from .wav file
        font_size: Font size for text overlay

    Returns:
        True if successful, False otherwise
    """
    import shutil
    import tempfile

    try:
        bvh_name = Path(bvh_file).stem
        output_path = f"{output_file}.mp4"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Get text data
        global_text, local_text = get_text_for_file(bvh_name, gen_texts) if gen_texts else (None, None)
        print(f"  Text data: global='{global_text}', local='{local_text}'")

        # Find audio file
        audio_path = find_audio_file(bvh_file) if add_audio else None
        print(f"  Audio file: {audio_path}")

        # Determine if we need post-processing
        needs_postprocessing = audio_path or (global_text or local_text)

        # Set output paths
        if needs_postprocessing:
            # Render to temp file first, then process to final output
            temp_video = os.path.join(tempfile.mkdtemp(), 'temp_render.mp4')
            final_output = output_path
        else:
            # Render directly to final output
            temp_video = output_path
            final_output = None

        # Parse BVH file
        parser = BVHParser()
        bvh_data = parser.parse(bvh_file)

        # Convert to position data
        pos_data = MocapParameterizer('position').fit_transform([bvh_data])

        # Render video to temp file
        print(f"Rendering: {bvh_file}")

        if view == "second":
            render_mp4_second_person(pos_data[0], temp_video, axis_scale=axis_scale)
        else:  # third person view
            render_mp4(pos_data[0], temp_video, axis_scale=axis_scale,
                      elev=elev, azim=azim, track_character=track_character)

        # Add audio and text overlay if needed
        if needs_postprocessing:
            print(f"Adding audio and text overlay...")
            success = add_audio_and_text_to_video(
                video_path=temp_video,
                audio_path=audio_path,
                global_text=global_text,
                local_text=local_text,
                output_path=final_output,
                font_size=font_size
            )
            # Clean up temp file
            temp_dir = os.path.dirname(temp_video)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return success

        return True

    except Exception as e:
        print(f"Error rendering {bvh_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_render_bvh(input_dir, output_dir="videos", gen_texts_json=None,
                     view="third", axis_scale=200, elev=45, azim=45,
                     track_character=False, add_audio=True, font_size=48):
    """
    Batch render all BVH files in a directory tree with optional audio and text overlay.

    Args:
        input_dir: Input directory containing BVH files
        output_dir: Output directory for rendered videos
        gen_texts_json: Path to gen_texts.json file or directory containing it (if None, auto-search in input_dir)
        view: Camera view - "third" for fixed third-person, "second" for second-person tracking
        axis_scale: Axis scale for the view
        elev: Elevation angle for third-person view
        azim: Azimuth angle for third-person view
        track_character: Whether to track character movement (third-person view only)
        add_audio: Whether to add audio from .wav files
        font_size: Font size for text overlay
    """
    # Load gen_texts.json
    gen_texts = {}
    if gen_texts_json:
        if os.path.isdir(gen_texts_json):
            json_path = find_gen_texts_json(gen_texts_json)
        else:
            json_path = Path(gen_texts_json)

        if json_path and json_path.exists():
            gen_texts = load_gen_texts(json_path)
            print(f"Loaded gen_texts.json with {len(gen_texts)} entries")
        else:
            print(f"Warning: gen_texts.json not found at {gen_texts_json}")
    else:
        # Auto-search gen_texts.json in input directory
        json_path = find_gen_texts_json(input_dir)
        if json_path and json_path.exists():
            gen_texts = load_gen_texts(json_path)
            print(f"Auto-loaded gen_texts.json with {len(gen_texts)} entries")
        else:
            print(f"No gen_texts.json found in {input_dir}, text overlay disabled")

    # Find all BVH files
    bvh_files = find_bvh_files(input_dir)

    if not bvh_files:
        print(f"No BVH files found in {input_dir}")
        return

    print(f"Found {len(bvh_files)} BVH files")

    # Process each file
    success_count = 0
    failed_files = []

    for bvh_file, relative_path in tqdm(bvh_files, desc="Rendering BVH files"):
        # Create output subdirectory structure
        output_subdir = Path(output_dir) / Path(relative_path).parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Generate output filename (same as input but without .bvh extension)
        bvh_name = Path(bvh_file).stem
        output_file = str(output_subdir / bvh_name)

        # Render the file
        success = render_single_bvh(
            bvh_file=bvh_file,
            output_file=output_file,
            gen_texts=gen_texts,
            view=view,
            axis_scale=axis_scale,
            elev=elev,
            azim=azim,
            track_character=track_character,
            add_audio=add_audio,
            font_size=font_size
        )

        if success:
            success_count += 1
        else:
            failed_files.append(bvh_file)

    # Print summary
    print(f"\nRendering complete!")
    print(f"Successfully processed: {success_count}/{len(bvh_files)} files")
    print(f"Output directory: {output_dir}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch render BVH files to videos with audio and text overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render all BVH files with audio and text (gen_texts.json in input directory)
  python utils/batch_render_bvh.py --input_dir ./bvh_files

  # Specify gen_texts.json location
  python utils/batch_render_bvh.py --input_dir ./bvh_files --gen_texts_json ./result/generated/local_2_clip/gen_texts.json

  # Render with second-person view
  python utils/batch_render_bvh.py --input_dir ./bvh_files --view second

  # Render with third-person view that tracks character movement
  python utils/batch_render_bvh.py --input_dir ./bvh_files --view third --track_character

  # Custom font size for text overlay
  python utils/batch_render_bvh.py --input_dir ./bvh_files --font_size 60
        """
    )

    parser.add_argument("--input_dir", "-i", required=True,
                        help="Input directory containing BVH files (will search recursively)")
    parser.add_argument("--output_dir", "-o", default="videos",
                        help="Output directory for rendered videos (default: videos)")
    parser.add_argument("--gen_texts_json", "-g", default=None,
                        help="Path to gen_texts.json file or directory containing it")
    parser.add_argument("--view", "-v", choices=["third", "second"], default="third",
                        help="Camera view: 'third' for fixed third-person, 'second' for second-person tracking (default: third)")
    parser.add_argument("--axis_scale", "-s", type=int, default=200,
                        help="Axis scale for the view (default: 200)")
    parser.add_argument("--elev", type=int, default=45,
                        help="Elevation angle for third-person view (default: 45)")
    parser.add_argument("--azim", type=int, default=45,
                        help="Azimuth angle for third-person view (default: 45)")
    parser.add_argument("--track_character", "-t", action="store_true",
                        help="Track character movement in third-person view (default: False)")
    parser.add_argument("--no_audio", action="store_true",
                        help="Do not add audio to videos")
    parser.add_argument("--font_size", type=int, default=48,
                        help="Font size for text overlay (default: 48)")

    args = parser.parse_args()

    batch_render_bvh(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gen_texts_json=args.gen_texts_json,
        view=args.view,
        axis_scale=args.axis_scale,
        elev=args.elev,
        azim=args.azim,
        track_character=args.track_character,
        add_audio=not args.no_audio,
        font_size=args.font_size
    )
