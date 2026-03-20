"""
Motion Dataset Loader with Text Conditioning

Dataset classes for loading motion data with text conditioning using CLIP encodings.
Supports both global and local text conditioning modes.

Usage:
    dataset = MotionDataset_Split_V2(
        data_root, datafiles_file, data_hparams, batch_size, random_drop
    )
    dataset.standardize(scalers)
    in_feats, text_feats, out_feats = dataset[index]
"""

import os
import random
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

from models.lgtm.text_encoder import CLIP_TextEncoder


def concat_dataframes(x, y):
    """Merge two dataframes synchronized by 'time' column."""
    if x.shape[0] < y.shape[0]:
        y = y[:x.shape[0]]
        y.index = x.index
    else:
        x = x[:y.shape[0]]
        x.index = y.index
    return pd.merge_asof(x, y, on='time', tolerance=pd.Timedelta('0.01s')).set_index('time')


def nans2zeros(x):
    """Convert NaN and Inf values to zeros."""
    ii = np.where(np.isinf(x))
    x[ii] = 0
    ii = np.where(np.isnan(x))
    x[ii] = 0
    return x


def dataframe_nansinf2zeros(df):
    """Replace NaN and Inf values in dataframe with zeros."""
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def align_start(x, y):
    """Truncate arrays to match minimum length."""
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]


def parse_token(f_name, inds):
    """Extract specific tokens from filename by indices."""
    basename = os.path.basename(f_name).split('.')[0]
    tokens = basename.split('_')
    out = ""
    assert len(inds) > 0
    for i in range(len(inds)):
        assert len(tokens) > inds[i], f"{inds[i]} out of range in {basename}"
        out += tokens[inds[i]]
        if i < len(inds) - 1:
            out += "_"
    return out


def styles2onehot(all_styles, style_token):
    """Convert style token to one-hot encoding."""
    oh = np.zeros(len(all_styles))
    for i in range(len(all_styles)):
        if style_token == all_styles[i]:
            oh[i] = 1
            return oh
    print("Style token error. Not found " + style_token)


def files_to_list(filename):
    """Read file paths from text file into list."""
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    return [f.rstrip() for f in files]


def get_json_data(data_root, text_folder, fname):
    """Load JSON text data for a given file."""
    fname = Path(fname).stem
    parts = fname.split('_')
    if parts[-1].startswith('mirrored'):
        parts = parts[:-2]
    else:
        parts = parts[:-1]
    json_prefix = '_'.join(parts)
    json_path = Path(data_root) / text_folder / f"{json_prefix}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Json file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resample_data(data, nframes_new, has_root_motion, mode='linear'):
    """Resample data to target frame count using interpolation."""
    nframes = data.shape[0]
    x = np.arange(0, nframes) / (nframes - 1)
    xnew = np.arange(0, nframes_new) / (nframes_new - 1)

    data_out = np.zeros((nframes_new, data.shape[1]))
    for jj in range(data.shape[1]):
        y = data[:, jj]
        f = interpolate.interp1d(x, y, bounds_error=False, kind=mode, fill_value='extrapolate')
        data_out[:, jj] = f(xnew)

    if has_root_motion:
        sc = nframes / nframes_new
        data_out[:, -3:] = data_out[:, -3:] * sc

    return torch.from_numpy(data_out).float()


class MotionDataset_Split_V2(torch.utils.data.Dataset):
    """Motion dataset with text conditioning for global/local training."""

    def __init__(self, data_root, datafiles_file, data_hparams=None, batch_size=None, random_drop=0.1):
        files = files_to_list(datafiles_file)

        self.timestretch_prob = data_hparams.get("timestretch_prob", 0.1)
        self.timestretch_factor = data_hparams.get("timestretch_factor", 0)
        self.segment_length = data_hparams["segment_length"]

        max_segment_length = int(self.segment_length * (1.0 + self.timestretch_factor))

        start_idx = 0
        data = {"input": [], "output": [], "styles": [], "text": [], "local_text": []}
        indexes = []

        Clip = CLIP_TextEncoder(data_hparams["clip_model_name"], output_dim=data_hparams["clip_embedding_dim"])

        random.shuffle(files)
        for fi in range(len(files)):
            fname = files[fi]
            if fname == "":
                continue

            # Load input features
            in_mod = data_hparams["input_modality"]
            indata_file = Path(data_root) / f'{fname}.{in_mod}.pkl'
            infeats_file = Path(data_root) / data_hparams["input_feats_file"]
            infeats_cols = np.loadtxt(infeats_file, dtype=str).tolist()
            with open(indata_file, 'rb') as f:
                in_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32'))
                in_times = pd.to_timedelta(in_feats.index)
                in_feats = in_feats[infeats_cols].values
                self.n_input = in_feats.shape[1]

            # Load output features
            out_mod = data_hparams["output_modality"]
            outdata_file = Path(data_root) / f'{fname}.{out_mod}.pkl'
            with open(outdata_file, 'rb') as f:
                out_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32'))
                out_times = pd.to_timedelta(out_feats.index)
                out_feats = out_feats.values
                self.n_output = out_feats.shape[1]

            in_feats, out_feats = align_start(in_feats, out_feats)
            in_times, out_times = align_start(in_times, out_times)

            trim_edges = data_hparams["trim_edges"]
            if trim_edges > 0:
                in_feats = in_feats[trim_edges: -trim_edges]
                out_feats = out_feats[trim_edges: -trim_edges]
                in_times = in_times[trim_edges: -trim_edges]
                out_times = out_times[trim_edges: -trim_edges]

            n_frames = in_feats.shape[0]

            # Load styles if available
            if "styles_file" in data_hparams:
                styles_file = Path(data_root) / data_hparams["styles_file"]
                all_styles = np.loadtxt(styles_file, dtype=str).tolist()
                styles_oh = np.tile(styles2onehot(all_styles, parse_token(files[fi], data_hparams["style_index"])), (n_frames, 1))
                self.n_styles = len(all_styles)
            else:
                self.n_styles = 0

            # Load text data
            try:
                text_data = get_json_data(data_root, data_hparams["text_folder"], fname)
            except FileNotFoundError as e:
                print(f"Missing json for {fname}: {e}")
                continue

            # Global text conditioning
            if data_hparams["train_stage"] == "global":
                global_data = text_data["global"]
                encoded_text = Clip.encode(global_data).repeat(n_frames, 1)

                seglen = max_segment_length
                if n_frames >= seglen:
                    idx_array = torch.arange(start_idx, start_idx + n_frames).unfold(0, seglen, 1)
                    data["input"].append(in_feats)
                    data["output"].append(out_feats)
                    if self.n_styles > 0:
                        data["styles"].append(styles_oh)
                    data["text"].append(encoded_text)
                    indexes.append(idx_array)
                    start_idx += n_frames

            # Local text conditioning
            elif data_hparams["train_stage"] == "local":
                global_data = text_data["global"]
                local_texts = text_data["local"]

                for local_text in local_texts:
                    start_time = pd.to_timedelta(local_text["start_time"]) - pd.Timedelta(seconds=0.05)
                    end_time = pd.to_timedelta(local_text["end_time"]) + pd.Timedelta(seconds=0.05)

                    mask = (in_times >= start_time) & (in_times <= end_time)

                    local_in_feats = in_feats[mask]
                    local_out_feats = out_feats[mask]
                    if self.n_styles > 0:
                        local_styles = styles_oh[mask]

                    if local_in_feats.shape[0] == 0:
                        continue

                    # Random drop to decouple global and local text
                    if random.random() > random_drop:
                        text_all = Clip.encode(global_data + "," + local_text["action"]).repeat(local_in_feats.shape[0], 1)
                    else:
                        text_all = Clip.encode(local_text["action"]).repeat(local_in_feats.shape[0], 1)

                    seglen = max_segment_length
                    if local_in_feats.shape[0] >= seglen:
                        idx_array = torch.arange(start_idx, start_idx + local_in_feats.shape[0]).unfold(0, seglen, 1)
                        data["input"].append(local_in_feats)
                        data["output"].append(local_out_feats)
                        if self.n_styles > 0:
                            data["styles"].append(local_styles)
                        data["text"].append(text_all)
                        indexes.append(idx_array)
                        start_idx += local_in_feats.shape[0]

        data["input"] = torch.from_numpy(np.vstack(data["input"])).float()
        data["output"] = torch.from_numpy(np.vstack(data["output"])).float()
        if self.n_styles > 0:
            data["styles"] = torch.from_numpy(np.vstack(data["styles"])).float()
        data["text"] = torch.vstack(data["text"]).cpu().float()

        print(data["input"].shape)
        print(data["output"].shape)
        if self.n_styles > 0:
            print(data["styles"].shape)
        print(data["text"].shape)
        print(f"=== total number of frames: {data['output'].shape[0]} =====")

        self.data = data
        self.mean_pose = data["output"].mean(axis=0)
        indexes = torch.cat(indexes, dim=0)
        self.indexes = indexes

    def assert_not_const(self, data):
        eps = 1e-6
        assert (data.std(axis=0) < eps).sum() == 0

    def fit_scalers(self):
        """Fit StandardScaler on input, output, and text data.

        Returns:
            dict: Dictionary containing in_scaler, out_scaler, text_scaler
        """
        in_scaler = StandardScaler()
        self.assert_not_const(self.data["input"])
        in_scaler.fit(self.data["input"])

        out_scaler = StandardScaler()
        self.assert_not_const(self.data["output"])
        out_scaler.fit(self.data["output"])

        text_scaler = StandardScaler()
        text_scaler.fit(self.data["text"])

        return {"in_scaler": in_scaler, "out_scaler": out_scaler, "text_scaler": text_scaler}

    def standardize(self, scalers):
        """Standardize data using fitted scalers."""
        self.data["input"] = torch.from_numpy(scalers["in_scaler"].transform(self.data["input"])).float()
        self.data["output"] = torch.from_numpy(scalers["out_scaler"].transform(self.data["output"])).float()
        self.data["text"] = torch.from_numpy(scalers["text_scaler"].transform(self.data["text"])).float()

    def timestretch(self, data, segment_length, factor, has_root_motion=False):
        """Apply time stretching/compression to data."""
        if factor < 1.0:
            return resample_data(data[:int(factor * segment_length)], segment_length, has_root_motion)
        elif factor > 1.0:
            return resample_data(data, int(factor * segment_length), has_root_motion)[:segment_length]
        else:
            return data[:segment_length]

    def __getitem__(self, index):
        """Get a sample from the dataset."""
        in_feats = self.data["input"][self.indexes[index]]
        out_feats = self.data["output"][self.indexes[index]]
        text_feats = self.data["text"][self.indexes[index]]

        if self.timestretch_factor > 0:
            if torch.rand((1,)) < self.timestretch_prob:
                segment_length = self.segment_length
                factor = torch.rand((1,)) * self.timestretch_factor * 2 - self.timestretch_factor + 1
                in_feats = self.timestretch(in_feats, segment_length, factor, has_root_motion=False)
                out_feats = self.timestretch(out_feats, segment_length, factor, has_root_motion=True)
                text_feats = self.timestretch(text_feats, segment_length, factor, has_root_motion=True)
            else:
                in_feats = in_feats[:self.segment_length]
                out_feats = out_feats[:self.segment_length]
                text_feats = text_feats[:self.segment_length]

        return (in_feats, text_feats, out_feats)

    def __len__(self):
        return self.indexes.size(0)


class MotionDataset_Split_Memory(torch.utils.data.Dataset):
    """Motion dataset with memory frames for conditioning."""

    def __init__(self, data_root, datafiles_file, data_hparams=None, batch_size=None):
        files = files_to_list(datafiles_file)

        self.timestretch_prob = data_hparams.get("timestretch_prob", 0.1)
        self.timestretch_factor = data_hparams.get("timestretch_factor", 0)
        self.segment_length = data_hparams["segment_length"]
        self.batch_size = batch_size

        max_segment_length = int(self.segment_length * (1.0 + self.timestretch_factor))
        self.memory_length = data_hparams.get("memory_length", 0)

        start_idx = 0
        data = {"input": [], "output": [], "styles": [], "text": []}
        indexes = []

        Clip = CLIP_TextEncoder(data_hparams["clip_model_name"], output_dim=data_hparams["clip_embedding_dim"])

        random.shuffle(files)
        for fi in range(len(files)):
            fname = files[fi]
            if fname == "":
                continue

            # Load input features
            in_mod = data_hparams["input_modality"]
            indata_file = Path(data_root) / f'{fname}.{in_mod}.pkl'
            infeats_file = Path(data_root) / data_hparams["input_feats_file"]
            infeats_cols = np.loadtxt(infeats_file, dtype=str).tolist()
            with open(indata_file, 'rb') as f:
                in_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32'))
                in_times = pd.to_timedelta(in_feats.index)
                in_feats = in_feats[infeats_cols].values
                self.n_input = in_feats.shape[1]

            # Load output features
            out_mod = data_hparams["output_modality"]
            outdata_file = Path(data_root) / f'{fname}.{out_mod}.pkl'
            with open(outdata_file, 'rb') as f:
                out_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32'))
                out_times = pd.to_timedelta(out_feats.index)
                out_feats = out_feats.values
                self.n_output = out_feats.shape[1]

            in_feats, out_feats = align_start(in_feats, out_feats)
            in_times, out_times = align_start(in_times, out_times)

            trim_edges = data_hparams["trim_edges"]
            if trim_edges > 0:
                in_feats = in_feats[trim_edges: -trim_edges]
                out_feats = out_feats[trim_edges: -trim_edges]
                in_times = in_times[trim_edges: -trim_edges]
                out_times = out_times[trim_edges: -trim_edges]

            n_frames = in_feats.shape[0]

            # Load styles if available
            if "styles_file" in data_hparams:
                styles_file = Path(data_root) / data_hparams["styles_file"]
                all_styles = np.loadtxt(styles_file, dtype=str).tolist()
                styles_oh = np.tile(styles2onehot(all_styles, parse_token(files[fi], data_hparams["style_index"])), (n_frames, 1))
                self.n_styles = len(all_styles)
            else:
                self.n_styles = 0

            # Load text data
            try:
                text_data = get_json_data(data_root, data_hparams["text_folder"], fname)
            except FileNotFoundError as e:
                print(f"Missing json for {fname}: {e}")
                continue

            # Global text conditioning
            if data_hparams["train_stage"] == "global":
                if self.batch_size is not None:
                    adjusted_n_frames = self._adjust_length_for_batch_size(n_frames, self.batch_size)
                    if adjusted_n_frames < n_frames:
                        print(f"Global file {fname}: adjusting frames from {n_frames} to {adjusted_n_frames}")
                        in_feats = in_feats[:adjusted_n_frames]
                        out_feats = out_feats[:adjusted_n_frames]
                        in_times = in_times[:adjusted_n_frames]
                        out_times = out_times[:adjusted_n_frames]
                        if self.n_styles > 0:
                            styles_oh = styles_oh[:adjusted_n_frames]
                        n_frames = adjusted_n_frames

                self.text_data = text_data["global"]
                self.text_encoder = Clip.encode(self.text_data).repeat(n_frames, 1)
                seglen = max_segment_length
                if n_frames >= seglen:
                    idx_array = torch.arange(start_idx, start_idx + n_frames).unfold(0, seglen, 1)
                    data["input"].append(in_feats)
                    data["output"].append(out_feats)
                    if self.n_styles > 0:
                        data["styles"].append(styles_oh)
                    data["text"].append(self.text_encoder)
                    indexes.append(idx_array)
                    start_idx += n_frames

            # Local text conditioning
            elif data_hparams["train_stage"] == "local":
                local_texts = text_data["local"]
                text_encodings = []

                for local_text in local_texts:
                    start_time = pd.to_timedelta(local_text["start_time"]) - pd.Timedelta(seconds=0.5)
                    end_time = pd.to_timedelta(local_text["end_time"]) + pd.Timedelta(seconds=0.5)

                    mask = (in_times >= start_time) & (in_times <= end_time)

                    local_in_feats = in_feats[mask]
                    local_out_feats = out_feats[mask]
                    if self.n_styles > 0:
                        local_styles = styles_oh[mask]

                    if local_in_feats.shape[0] == 0:
                        continue

                    encoded_text = Clip.encode(local_text["action"]).repeat(local_in_feats.shape[0], 1)
                    text_encodings.append(encoded_text)

                    seglen = max_segment_length
                    if local_in_feats.shape[0] >= seglen:
                        idx_array = torch.arange(start_idx, start_idx + local_in_feats.shape[0]).unfold(0, seglen, 1)
                        data["input"].append(local_in_feats)
                        data["output"].append(local_out_feats)
                        if self.n_styles > 0:
                            data["styles"].append(local_styles)
                        data["text"].append(encoded_text)
                        indexes.append(idx_array)
                        start_idx += local_in_feats.shape[0]

                if text_encodings:
                    self.text_encoder = torch.cat(text_encodings, dim=0)

        data["input"] = torch.from_numpy(np.vstack(data["input"])).float()
        data["output"] = torch.from_numpy(np.vstack(data["output"])).float()
        if self.n_styles > 0:
            data["styles"] = torch.from_numpy(np.vstack(data["styles"])).float()
        data["text"] = torch.vstack(data["text"]).cpu().float()

        print(data["input"].shape)
        print(data["output"].shape)
        print(data["styles"].shape)
        print(data["text"].shape)
        print(f"=== total number of frames: {data['output'].shape[0]} =====")

        self.data = data
        self.mean_pose = data["output"].mean(axis=0)
        indexes = torch.cat(indexes, dim=0)
        self.indexes = indexes

    def _adjust_length_for_batch_size(self, length, batch_size):
        """Adjust length to be a multiple of batch_size."""
        if batch_size is None or batch_size <= 0:
            return length
        return (length // batch_size) * batch_size

    def assert_not_const(self, data):
        eps = 1e-6
        assert (data.std(axis=0) < eps).sum() == 0

    def fit_scalers(self):
        """Fit StandardScaler on input, output, and text data.

        Returns:
            dict: Dictionary containing in_scaler, out_scaler, text_scaler
        """
        in_scaler = StandardScaler()
        self.assert_not_const(self.data["input"])
        in_scaler.fit(self.data["input"])

        out_scaler = StandardScaler()
        self.assert_not_const(self.data["output"])
        out_scaler.fit(self.data["output"])

        text_scaler = StandardScaler()
        text_scaler.fit(self.data["text"])

        return {"in_scaler": in_scaler, "out_scaler": out_scaler, "text_scaler": text_scaler}

    def standardize(self, scalers):
        """Standardize data using fitted scalers."""
        self.data["input"] = torch.from_numpy(scalers["in_scaler"].transform(self.data["input"])).float()
        self.data["output"] = torch.from_numpy(scalers["out_scaler"].transform(self.data["output"])).float()
        self.data["text"] = torch.from_numpy(scalers["text_scaler"].transform(self.data["text"])).float()

    def timestretch(self, data, segment_length, factor, has_root_motion=False):
        """Apply time stretching/compression to data."""
        if factor < 1.0:
            return resample_data(data[:int(factor * segment_length)], segment_length, has_root_motion)
        elif factor > 1.0:
            return resample_data(data, int(factor * segment_length), has_root_motion)[:segment_length]
        else:
            return data[:segment_length]

    def __getitem__(self, index):
        """Get a sample from the dataset with memory conditioning."""
        full_in_feats = self.data["input"][self.indexes[index]]
        full_out_feats = self.data["output"][self.indexes[index]]
        text_feats = self.data["text"][self.indexes[index]]

        if self.timestretch_factor > 0:
            if torch.rand((1,)) < self.timestretch_prob:
                segment_length = self.segment_length
                factor = torch.rand((1,)) * self.timestretch_factor * 2 - self.timestretch_factor + 1
                full_in_feats = self.timestretch(full_in_feats, segment_length, factor, has_root_motion=False)
                full_out_feats = self.timestretch(full_out_feats, segment_length, factor, has_root_motion=True)
                text_feats = self.timestretch(text_feats, segment_length, factor, has_root_motion=True)
            else:
                full_in_feats = full_in_feats[:self.segment_length]
                full_out_feats = full_out_feats[:self.segment_length]
                text_feats = text_feats[:self.segment_length]

        memory_frames = self.memory_length
        memory_motion = full_out_feats[:memory_frames]

        output_in = full_in_feats[memory_frames:]
        output_out = full_out_feats[memory_frames:]
        output_text = text_feats[memory_frames:] if text_feats is not None else None

        return (output_in, output_text, output_out, memory_motion)

    def __len__(self):
        return len(self.indexes)


if __name__ == "__main__":
    from utils.hparams import get_hparams

    hparams, conf_name = get_hparams()
    print(hparams)
    print(conf_name)
