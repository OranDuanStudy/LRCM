"""
Logging Mixin for Motion Prediction Results

Provides logging functionality for motion prediction results including:
- BVH file export
- Video rendering from motion data
- Jerk metric calculation

Usage:
    Inherit from LoggingMixin in PyTorch Lightning module to use:
    - log_results(pred_clips, file_name, log_prefix, logdir, render_video, view)
    - log_jerk(x, log_prefix)
"""

import io
import json
from pathlib import Path
import joblib as jl
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.pipeline import Pipeline
from pymo.writers import BVHWriter
from pymo.data import MocapData
from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4, render_mp4_second_person


class LoggingMixin:
    """Mixin class for logging motion prediction results."""

    def log_results(self, pred_clips, file_name, log_prefix, logdir=None, render_video=True, view=None, track_character=False):
        """
        Log prediction results, save predicted clips as BVH files, and optionally render as video.

        Parameters:
        pred_clips: Predicted clips in time series format.
        file_name: Base name for saving files.
        log_prefix: Log prefix used for file naming.
        logdir: Directory to save logs, defaults to None which uses the default log directory.
        render_video: Whether to render video, defaults to True.
        view: View type, None or "third" for third-person fixed view, "second" for second-person tracking view.
        track_character: Whether to track character movement (keeps fixed elev and azim, camera follows character), defaults to False.
        """

        # Set log directory, defaults to the preset save directory from class logger
        if logdir is None:
            logdir = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"

        # If log_prefix is not empty, append it to file_name
        if len(log_prefix.strip())>0:
            file_name = file_name + "_" + log_prefix

        # Convert predicted clips to BVH format data
        bvh_data = self.feats_to_bvh(pred_clips)
        nclips = len(bvh_data)
        framerate = np.rint(1/bvh_data[0].framerate)
        # print('framerate: \n', framerate)
        # print('nclips: \n', nclips)

        # If maximum render clips is configured, limit the number of clips to render
        if self.hparams.Validation["max_render_clips"]:
            nclips = min(nclips, self.hparams.Validation["max_render_clips"])

        # Write BVH files
        self.write_bvh(bvh_data[:nclips], log_dir=logdir, name_prefix=file_name)

        # If video rendering is requested, convert BVH data to position data
        if render_video:
            pos_data = self.bvh_to_pos(bvh_data)
            self.render_video(pos_data[:nclips], log_dir=logdir, name_prefix=file_name, view=view, track_character=track_character)
                   

    def feats_to_bvh(self, pred_clips):
        """Convert feature predictions to BVH format.

        Args:
            pred_clips: Predicted feature clips.

        Returns:
            BVH data objects.
        """
        data_pipeline = jl.load(Path(self.hparams.dataset_root) / self.hparams.Data["datapipe_filename"])
        n_feats = data_pipeline["cnt"].n_features
        data_pipeline["root"].separate_root = False
        bvh_data = data_pipeline.inverse_transform(pred_clips[:, :, :n_feats])
        return bvh_data

    def write_bvh(self, bvh_data, log_dir="", name_prefix=""):
        """Write BVH data to files.

        Args:
            bvh_data: List of BVH data objects.
            log_dir: Output directory.
            name_prefix: File name prefix.
        """
        writer = BVHWriter()
        nclips = len(bvh_data)
        for i in range(nclips):
            if nclips > 1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}.bvh"
            else:
                fname = f"{log_dir}/{name_prefix}.bvh"
            print('writing:' + fname)
            with open(fname, 'w') as f:
                writer.write(bvh_data[i], f)

    def bvh_to_pos(self, bvh_data):
        """Convert BVH data to joint positions.

        Args:
            bvh_data: BVH data objects.

        Returns:
            Position data arrays.
        """
        return MocapParameterizer('position').fit_transform(bvh_data)

    def render_video(self, pos_data, log_dir="", name_prefix="", view=None, track_character=False):
        # write bvh and skeleton motion
        nclips = len(pos_data)
        for i in range(nclips):
            if nclips>1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}"
            else:
                fname = f"{log_dir}/{name_prefix}"
            print('writing:' + fname + ".mp4")
            if view==None or view=="third":
                render_mp4(pos_data[i], fname + ".mp4", axis_scale=200, track_character=track_character)
            elif view=="second":
                render_mp4_second_person(pos_data[i], fname + ".mp4", axis_scale=30)
        
            

    def log_jerk(self, x, log_prefix):
        """Calculate and log jerk metric (rate of change of acceleration).

        Args:
            x: Motion data tensor.
            log_prefix: Prefix for logging metric name.
        """
        deriv = x[:, 1:] - x[:, :-1]
        acc = deriv[:, 1:] - deriv[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        self.log(f'{log_prefix}_jerk', torch.mean(torch.abs(jerk)))
