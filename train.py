"""
stage1:
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py --dataset_root data/Multimodel_Text_dataset_updating --hparams_file ./hparams/Mamba_dance_stage1.yaml --ckpt_file None
stage2:
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py --dataset_root data/Multimodel_Text_dataset_updating --hparams_file ./hparams/Mamba_dance_stage2.yaml --ckpt_file ./pretrained_models/dance_LRCM_stage1.ckpt
stage3:
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py --dataset_root data/Multimodel_Text_dataset_updating --hparams_file ./hparams/Mamba_dance_stage3.yaml --ckpt_file ./pretrained_models/dance_LRCM_stage2.ckpt
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelPruning
from torch import quantization

from sklearn.preprocessing import StandardScaler

from utils.motion_dataset import MotionDataset_Split_V2, MotionDataset_Split_Memory
from models.LightningModel import LitLRCM
from utils.hparams import get_hparams
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def data_loader(dataset_root, file_name, data_hparams, batch_size, num_workers=16, shuffle=True):
    """
    Create a DataLoader for motion dataset.

    Args:
        dataset_root: Root directory of the dataset
        file_name: Name of the data file
        data_hparams: Data hyperparameters containing memory_length and other settings
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader: Configured data loader for motion dataset
    """
    print("dataset_root: " + dataset_root)

    # Select dataset class based on memory configuration
    if data_hparams["memory_length"] > 0 and data_hparams.get("use_memory_dataset", False):
        print(f"Using MotionDataset_Split_Memory with memory_length={data_hparams['memory_length']}")
        dataset = MotionDataset_Split_Memory(
            dataset_root,
            Path(dataset_root) / file_name,
            data_hparams=data_hparams,
            batch_size=batch_size
        )
    else:
        dataset = MotionDataset_Split_V2(
            dataset_root,
            Path(dataset_root) / file_name,
            data_hparams=data_hparams,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True,
        persistent_workers=True,
        collate_fn=None
    )


def dataloaders(dataset_root, data_hparams, batch_size, num_workers):
    """
    Create train, validation, and test dataloaders.

    Args:
        dataset_root: Root directory of the dataset
        data_hparams: Data hyperparameters
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    print("Loading data")
    print("train_files: ")
    train_dl = data_loader(dataset_root, data_hparams["traindata_filename"], hparams.Data, batch_size, num_workers, shuffle=True)
    print("val_files: ")
    val_dl = data_loader(dataset_root, data_hparams["testdata_filename"], hparams.Data, batch_size, num_workers, shuffle=False)
    print("test_files: ")
    test_dl = data_loader(dataset_root, data_hparams["testdata_filename"], hparams.Data, batch_size, num_workers, shuffle=False)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":

    # Load hyperparameters from config file
    hparams, conf_name = get_hparams()
    print(hparams)
    print(conf_name)

    # Verify dataset root exists
    assert os.path.exists(
        hparams.dataset_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    # Create dataloaders
    train_dl, val_dl, test_dl = dataloaders(hparams.dataset_root, hparams.Data, hparams.batch_size, hparams.num_dataloader_workers)


    # Resume from checkpoint or create new model
    if hparams.ckpt_file != 'None':
        ckpt = hparams.ckpt_file
        print(f"resuming from checkpoint: {ckpt}")

        scalers = train_dl.dataset.fit_scalers()
        hparams.Data["scalers"] = scalers
        print(hparams.Data)

        model = LitLRCM(hparams)

        checkpoint = torch.load(ckpt)
        print("Reusing the scalers from previous model.")
        model.load_state_dict(checkpoint['state_dict'], strict=False)


    else:
        # Create new model
        print("Fitting scalers")
        scalers = train_dl.dataset.fit_scalers()

        print("Setting scalers to model hparams")
        hparams.Data["scalers"] = scalers
        print(hparams.Data)

        print("Create model")
        model = LitLRCM(hparams)

    # Standardize data using fitted scalers
    print("Standardize data")
    train_dl.dataset.standardize(scalers)
    val_dl.dataset.standardize(scalers)
    test_dl.dataset.standardize(scalers)

    # Set mean pose for the model
    mean_pose = (train_dl.dataset.mean_pose + val_dl.dataset.mean_pose + test_dl.dataset.mean_pose) / 3
    print("  --->>> mean_pose.shape:", mean_pose.shape)
    model.mean_pose = mean_pose
    print(f"Set mean_pose after loading weights, shape: {mean_pose.shape}")

    # Setup trainer
    trainer_params = vars(hparams).copy()

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Model checkpoint callback - saves top 10 models based on training loss
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=10,
        monitor='Loss/train',
        mode='min'
    )

    # Initialize trainer with callbacks
    callbacks = [lr_monitor, checkpoint_callback]
    trainer = Trainer(callbacks=callbacks,**(trainer_params["Trainer"]), log_every_n_steps=1)


    # Start training
    print("Start training!")
    trainer.fit(model, train_dl, val_dl)
