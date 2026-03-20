"""
Hyperparameter Configuration Loader

Load hyperparameter configuration from JSON or YAML files.
Command line arguments can override config file values.

Usage:
    hparams, conf_name = get_hparams()

Args:
    --dataset_root: Root directory of the dataset
    --hparams_file: Path to hyperparameters file (.json or .yaml)
    --ckpt_file: Path to checkpoint file for fine-tuning
"""

import os
from argparse import ArgumentParser, Namespace

import json
import yaml


def get_hparams():
    """Load hyperparameter configuration from file and command line args.

    Returns:
        hparams: Namespace object with final hyperparameter configuration
        conf_name: Configuration file name (without path)
    """
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", help="Root directory of the dataset")
    parser.add_argument("--hparams_file", help="Path to the hyperparameters file")
    parser.add_argument("--ckpt_file", help="Path to checkpoint file for fine tuning")
    args = parser.parse_args()

    conf_name = os.path.basename(args.hparams_file)

    if args.hparams_file.endswith(".json"):
        with open(args.hparams_file, 'r') as f:
            hparams_json = json.load(f)
    elif args.hparams_file.endswith(".yaml"):
        with open(args.hparams_file, 'r') as f:
            hparams_json = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Please use .json or .yaml")

    params = vars(args)
    params.update(hparams_json)

    return Namespace(**params), conf_name
