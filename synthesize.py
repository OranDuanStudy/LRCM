from os.path import join
import copy
import os, sys, getopt
import torch
import numpy as np
import pickle as pkl
from pytorch_lightning import seed_everything
from utils.motion_dataset import styles2onehot, nans2zeros
from models.LightningModel import LitLRCM
from memory_profiler import profile
from tqdm import tqdm

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
torch.serialization.add_safe_globals({'StandardScaler': StandardScaler})

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lgtm.text_encoder import CLIP_TextEncoder

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
import math
torch.cuda.empty_cache()



def add_text_to_video(dest_dir, outfile, input_text):
    """
    Add text overlay to an already generated video.

    Args:
        dest_dir: Directory where the video file is located
        outfile: Output file name (without extension)
        input_text: Text to add
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        # Build input video path
        input_path = f"{dest_dir}/{outfile}.mp4"
        if not os.path.exists(input_path):
            print(f"Video file does not exist: {input_path}")
            return

        # Build output video path
        output_path = f"{dest_dir}/{outfile}_with_text.mp4"

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {input_path}")
            return

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Combine text
        combined_text = input_text

        print(f"Adding text to video: '{combined_text}'")

        # Try to load font
        try:
            base_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except:
            try:
                base_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
            except:
                base_font = ImageFont.load_default()

        # Create a temporary image to measure text
        temp_img = Image.new('RGB', (width, height))
        temp_draw = ImageDraw.Draw(temp_img)

        # Auto-wrap function
        def wrap_text(text, font, max_width):
            """Wrap text to fit within max width"""
            words = text.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                line = ' '.join(current_line)
                bbox = temp_draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]

                if line_width > max_width:
                    current_line.pop()
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

            return lines if lines else [text]

        # Maximum text width (90% of video width, leaving margin)
        max_text_width = int(width * 0.9)

        # Wrap text
        text_lines = wrap_text(combined_text, base_font, max_text_width)

        frame_count = 0
        margin = 20
        line_height = 35  # Line height

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to PIL image for better text rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Draw each line of text
            y_offset = margin
            for line in text_lines:
                # Calculate current line text size
                bbox = draw.textbbox((0, 0), line, font=base_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Text position (top-right aligned)
                text_x = width - text_width - margin
                text_y = y_offset

                # Draw background rectangle (enhance contrast)
                padding = 8
                bg_rect = [
                    (max(0, text_x - padding), text_y - padding),
                    (width, text_y + text_height + padding)
                ]
                draw.rectangle(bg_rect, fill=(0, 0, 0, 200))  # Semi-transparent black background

                # Draw text (white)
                draw.text((text_x, text_y), line, font=base_font, fill=(255, 255, 255))

                y_offset += line_height

            # Convert back to OpenCV format
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ")

        # Release resources
        cap.release()
        out.release()

        print(f"Text added to video and saved to: {output_path}")

        # Replace original file
        import shutil
        shutil.move(output_path, input_path)
        print(f"Original video replaced with text version: {input_path}")

    except ImportError:
        print("please install opencv-python Pillow")
        print("run: pip install opencv-python Pillow")
    except Exception as e:
        print(f"Error adding text to video: {e}")

def sample_mixmodels(models, batches, guidance_factors, memory_motion=None):
    """
    Sample animation frames using a mixture of models.

    Args:
        models: List of diffusion models with compatible noise schedules
        batches: List of batch data (local_cond, global_cond, extra_data)
        guidance_factors: Guidance factors for mixing model predictions
        memory_motion: Memory motion features for temporal consistency

    Returns:
        anim_clip: Generated animation frames as numpy array
        memory_motion: Updated memory for next segment
    """
    assert len(guidance_factors)==(len(models)-1), "n_guidance_factors should be eq to n_models-1"

    noise_sched_0 = models[0].noise_schedule

    o_scaler_0 = models[0].hparams["Data"]["scalers"]["out_scaler"]

    eps = 0.000001
    for i in range(1, len(models)):
        assert torch.all(torch.abs(models[i].noise_schedule - noise_sched_0)<eps), "different noise-schedule"
        o_scaler_i = models[i].hparams["Data"]["scalers"]["out_scaler"]
        assert np.all(np.abs(o_scaler_i.mean_-o_scaler_0.mean_)<eps), "different pose standardization"
        assert np.all(np.abs(o_scaler_i.scale_-o_scaler_0.scale_)<eps), "different pose standardization"

    beta = noise_sched_0.detach().cpu().numpy()
    talpha = 1 - beta
    talpha_cum = np.cumprod(talpha)
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = np.arange(0,len(beta), dtype=np.float32)

    ctrl, _, _ = batches[-1]
    poses = torch.randn(ctrl.shape[0], ctrl.shape[1], models[0].pose_dim, device=models[0].device)

    nbatch = poses.size(0)
    noise_scale = torch.from_numpy(alpha_cum**0.5).type_as(poses).unsqueeze(1)

    for n in tqdm(range(len(alpha) - 1, -1, -1), desc="Synthesizing"):
        c1 = 1 / alpha[n]**0.5
        c2 = beta[n] / (1 - alpha_cum[n])**0.5

        diffs = []
        with torch.amp.autocast(device_type='cuda'):
            for i, model in enumerate(models):
                l_cond, g_cond, _ = batches[i]
                diffs.append(model.diffusion_model_cross(poses, l_cond, g_cond, torch.tensor([T[n]], device=poses.device), memory_motion).squeeze(1))

        diff0=diffs[0]
        diff=diff0
        for i in range(len(guidance_factors)):
            diff += guidance_factors[i]*(diffs[i+1] - diff0)

        poses = c1 * (poses - c2 * diff)

        if n > 0:
            noise = torch.randn_like(poses)
            sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
            poses += sigma * noise

    out_poses = models[-1].destandardizeOutput(poses)
    if not models[-1].unconditional:
        out_ctrl = models[-1].destandardizeInput(ctrl)
        print("out_poses", out_poses.shape)
        print("out_ctrl", out_ctrl.shape)
        anim_clip = torch.cat((out_poses, out_ctrl), dim=2).cpu().detach().numpy()
    else:
        anim_clip = out_poses.cpu().detach().numpy()

    memory_motion = out_poses[:, -models[-1].memory_length: , : ] if models[-1].memory_length is not None else None

    return anim_clip, memory_motion


def do_synthesize(models, l_conds, g_conds, file_name, postfix,
                   trim, dest_dir, guidance_factors, gpu, render_video, outfile, segment_frames):
    """
    Main synthesis function that processes long sequences in segments.

    Args:
        models: List of diffusion models
        l_conds: List of local conditions (audio features)
        g_conds: List of global conditions (text embeddings)
        file_name: Output file name prefix
        postfix: Additional suffix for output file
        trim: Number of frames to trim from start/end
        dest_dir: Output directory
        guidance_factors: Guidance factors for model mixing
        gpu: Device to use (e.g., 'cuda:0')
        render_video: Whether to render output video
        outfile: Output file name
        segment_frames: Number of frames per segment
    """
    total_frames = l_conds[-1].size(1)
    times = math.ceil(total_frames / segment_frames)

    device = torch.device(gpu)
    all_clips = []

    memory_motion = None

    for t in tqdm(range(times), desc="Processing segment clips"):

        if t == 0:
            if models[-1].memory_length is not None and models[-1].memory_length > 0:
                memory_length = models[-1].memory_length

                if models[-1].mean_pose is not None:
                    mean_pose = models[-1].mean_pose
                    if isinstance(mean_pose, np.ndarray):
                        mean_pose = torch.from_numpy(mean_pose).float()
                    mean_pose = mean_pose.unsqueeze(0).unsqueeze(0)
                    memory_motion = mean_pose.repeat(l_conds[-1].size(0), memory_length, 1).to(device)
                else:
                    memory_motion = torch.zeros(l_conds[-1].size(0), memory_length, models[-1].pose_dim).to(device)
            else:
                memory_motion = None

        else:
            if models[-1].memory_length is not None and models[-1].memory_length > 0:
                memory_length = models[-1].memory_length

                prev_clip = torch.from_numpy(all_clips[-1]).to(device)
                memory_motion = prev_clip[:, -memory_length:, :].to(device) if memory_motion is None else memory_motion
            else:
                memory_motion = None

        memory_motion = models[-1].standardizeOutput(memory_motion).to(device) if memory_motion is not None else None

        seg_batches = []

        for i in range(len(models)):
            models[i].to(device)
            models[i].eval()

            start_frame = t * segment_frames
            end_frame = min((t + 1) * segment_frames, total_frames)

            l_cond = l_conds[i][:, start_frame:end_frame, :]
            g_cond = g_conds[i][:, start_frame:end_frame, :]

            batch = l_cond.to(device) if len(l_cond) > 0 else [], \
                    g_cond.to(device) if len(g_cond) > 0 else [], \
                    None
            seg_batches.append(batch)


        with torch.no_grad():
            clips, memory_pose = sample_mixmodels(models,
                                      seg_batches,
                                      guidance_factors,
                                      memory_motion)
            all_clips.append(clips)
            memory_motion = copy.deepcopy(memory_pose)
        del clips, memory_pose
        torch.cuda.empty_cache()

    full_clip = np.concatenate(all_clips, axis=1)
    models[-1].log_results(full_clip[:, trim : total_frames-trim, :],
                           outfile,
                           "",
                           logdir=dest_dir,
                           render_video=render_video,
                           view=None)


def nans2zeros(x):
    """
    Replace inf and NaN values with zeros.

    Args:
        x: numpy array or torch tensor

    Returns:
        Processed array with inf/NaN replaced by zeros
    """
    x_copy = x.copy() if isinstance(x, np.ndarray) else x.clone()

    if isinstance(x_copy, np.ndarray):
        x_copy[np.isinf(x_copy)] = 0
        x_copy[np.isnan(x_copy)] = 0
    else:
        x_copy[torch.isinf(x_copy)] = 0
        x_copy[torch.isnan(x_copy)] = 0

    return x_copy


def get_style_vector(styles_file, style_token, nbatch, nframes):
    """
    Generate style vector from style file and token.

    Args:
        styles_file: Path to file containing style labels
        style_token: Style token identifier
        nbatch: Batch size
        nframes: Number of frames

    Returns:
        Style vector of shape (nbatch, nframes, style_dim)
    """
    all_styles = np.loadtxt(styles_file, dtype=str)
    styles_onehot = styles2onehot(all_styles, style_token)
    styles = styles_onehot.repeat(nbatch, nframes,1)


def get_cond(model, data_dir, input_file, text_input, length, startframe=0, endframe=None):
    """
    Prepare condition data for model input.

    Args:
        model: Model containing hyperparameters
        data_dir: Data directory path
        input_file: Input feature file name
        text_input: Text description for conditioning
        length: Control sequence length
        startframe: Starting frame index
        endframe: Ending frame index

    Returns:
        tuple: (standardized_control, standardized_text)
    """
    with open(join(data_dir, input_file), 'rb') as f:
        ctrl = pkl.load(f)
    ctrl = ctrl[startframe:]

    if endframe>0 and endframe<ctrl.shape[0]:
        ctrl = ctrl[:endframe]

    input_feats_file = os.path.join(data_dir, model.hparams.Data["input_feats_file"])
    input_feats = np.loadtxt(input_feats_file, dtype=str)
    ctrl = ctrl[input_feats]
    ctrl = nans2zeros(torch.from_numpy(ctrl.values).float().unsqueeze(0))

    nbatch = ctrl.size(0)
    nframes = ctrl.size(1)

    Clip = CLIP_TextEncoder('ViT-B/32')

    if "text_modality" in model.hparams.Data:
        text_clip = Clip.encode(text_input)
        texts = text_clip.repeat(nbatch, nframes,1)


    return model.standardizeInput(ctrl), model.standardizeClIP_Input(texts)


def arg2tokens(arg, delim=","):
    """Split argument string by delimiter into list of tokens."""
    return arg.strip().split(delim)


def arg2tokens_f(arg, delim=","):
    """
    Convert string argument to list of floats.

    Args:
        arg: Input string to split
        delim: Delimiter character (default: ",")

    Returns:
        List of float values
    """
    ts=arg2tokens(arg, delim)
    out=[]
    for t in ts:
        out.append(float(t))
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hc:x:d:f:s:e:t:r:p:g:k:v:o:m:", [
            "checkpoints=", "data_dirs=", "input_files=", "input_text=",
            "start=", "end=", "trim=", "seed=", "postfix=", "dest_dir=",
            "gf=", "gpu=", "video=", "outfile=", "segment-frames="
        ])
    except getopt.GetoptError:
        print('python synthesize.py -c checkpoint -d data_dir -i input_file -t input_text -b start -e end -r seed -p postfix -l dest_dir -g gf -k gpu -v video -o outfile -m segment_frames')
        print('example usage: python synthesize.py --checkpoint=results/moglow/styleloco/lightning_logs/version_9/checkpoints/XXXXX.ckpt --data_dir=data/motorica/locomotion/processed_sm6_6/ --input_file=data/motorica/locomotion/processed_sm6_6/loco_act01_male_w65_h178_earth_ex05_mix_q03_2022-02-02_001.expmap_20fps.pkl --style=act01_earth --end=200 --model=moglow --seed=seed --segment-frames=300')
        sys.exit(2)

    # Default parameters
    trim = 0
    postfix=""
    dest_dir="results"
    seed=42
    startframe=0
    guidance_factors = []
    gpu="cuda:0"
    style_tokens=None
    render_video=True
    outfile=""
    segment_frames = 300

    # Parse command line arguments
    for opt, arg in opts:
        if opt == '-h':
            print ('python synthesize.py -c checkpoint -d data_dir -i input_file -t input_text -b start -e end -r seed -p postfix -l dest_dir -g gf -k gpu -v video -o outfile -m segment_frames')
            print ('example usage: python synthesize.py --checkpoints=checkpoint.ckpt --data_dirs=data_dir --input_files=input_file.pkl --input_text="text prompt" --start=0 --end=300 --seed=42 --postfix=0 --trim=0 --dest_dir=results --gpu=cuda:0 --video=true --outfile=output --segment-frames=300')
            sys.exit()
        elif opt in ("-c", "--checkpoints"):
            checkpoints = arg2tokens(arg)
        elif opt in ("-d", "--data_dirs"):
            data_dirs = arg2tokens(arg)
        elif opt in ("-f", "--input_files"):
            input_files = arg2tokens(arg)
        elif opt in ("-t", "--input_text"):
            input_text = arg
        elif opt in ("-b", "--start"):
            startframe = int(arg)
        elif opt in ("-e", "--end"):
            endframe = int(arg)
        elif opt in ("-g", "--gf"):
            guidance_factors = arg2tokens_f(arg)
        elif opt in ("-t", "--trim"):
            trim = int(arg)
        elif opt in ("-r", "--seed"):
            seed = int(arg)
        elif opt in ("-p", "--postfix"):
            postfix = arg
        elif opt in ("-l", "--dest_dir"):
            dest_dir = arg
        elif opt in ("-k", "--gpu"):
            gpu = arg
        elif opt in ("-v", "--video"):
            render_video = arg.lower()=="true"
        elif opt in ("-o", "--outfile"):
            outfile = arg
        elif opt in ("-m", "--segment-frames"):
            segment_frames = int(arg)

    out_file_name = os.path.basename(input_files[0]).split('.')[0]
    seed_everything(seed)

    models = []
    l_conds = []
    g_conds = []

    print("text:", input_text)
    for i in range(len(checkpoints)):
        model = LitLRCM.load_from_checkpoint(checkpoints[i],dataset_root=data_dirs[i], strict=False)

        checkpoint = torch.load(checkpoints[i], map_location='cpu')
        if 'state_dict' in checkpoint and 'mean_pose' in checkpoint['state_dict']:
            model.register_buffer('mean_pose', checkpoint['state_dict']['mean_pose'])
            print(f"Successfully loaded mean_pose from checkpoint, shape: {model.mean_pose.shape}")

        models.append(model)

        if input_text is not None:
            l_cond, text = get_cond(model, data_dirs[i], input_files[i], input_text, endframe, startframe, endframe)
        else:
            l_cond, text = get_cond(model, data_dirs[i], input_files[i], "", endframe, startframe, endframe)


        l_conds.append(l_cond)
        g_conds.append(text)

    # Run synthesis
    do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile, segment_frames)

    # Add text overlay to video
    if render_video:
        add_text_to_video(dest_dir, outfile, input_text)
