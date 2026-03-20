import os
import sys

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from models.BaseModel import BaseModel

from models.nn import CrossResdualBlock
from models.mamba.mambamotion import MultiFeaturesMemoryBlock

from models.nn import LRCM

from typing import Tuple, Optional, Union, Dict
from argparse import Namespace

from models.lgtm.text_encoder import CLIP_TextEncoder

from tqdm import tqdm
from torchinfo import summary
from einops import rearrange, repeat

import random


class LitLRCM(BaseModel):
    def __init__(self, conf, **kwargs):
        super().__init__(conf)

        self.ROBUST_TRAINING_NOISE_RATIO = 0.12
        self.NOISE_ADD_RATIO = 0.06

        self.input_dim = 0
        self.style_dim = 0

        if self.hparams.Data["scalers"]["in_scaler"] is not None:
            self.input_dim = self.hparams.Data["scalers"]["in_scaler"].mean_.shape[0]

        self.pose_dim = self.hparams.Data["scalers"]["out_scaler"].mean_.shape[0]

        self.text_dim = self.hparams.Data["scalers"]["text_scaler"].mean_.shape[0]
        self.clip_dim = self.text_dim

        self.unconditional = self.input_dim == 0

        self.audio_dim = self.input_dim

        self.seg_length = self.hparams.Data["segment_length"]
        diff_params = self.hparams.Diffusion

        beta_min = diff_params["noise_schedule_start"]
        beta_max = diff_params["noise_schedule_end"]
        self.n_noise_schedule = diff_params["n_noise_schedule"]

        self.noise_schedule_name = "linear"
        self.noise_schedule = torch.linspace(beta_min, beta_max, self.n_noise_schedule)
        self.noise_level = torch.cumprod(1 - self.noise_schedule, dim=0)

        nn_name = diff_params["name"]
        nn_args = diff_params["args"][nn_name]

        self.training_stage = self.hparams["training_stage"]

        self.memory_parms = self.hparams.DualMambaMotionMemory
        self.memory_length = self.hparams.Data["memory_length"] if self.memory_parms["use_DMMM"] == True else None

        self.diffusion_model_cross = LRCM(self.pose_dim,
                                         self.hparams.Diffusion["residual_layers"],
                                         self.hparams.Diffusion["residual_channels"],
                                         self.hparams.Diffusion["embedding_dim"],
                                         self.audio_dim,
                                         self.clip_dim,
                                         self.n_noise_schedule,
                                         nn_name,
                                         self.memory_parms,
                                         nn_args)

        self.loss_fn = nn.MSELoss()
        self.loss_motion = nn.MSELoss()

        self.register_buffer('mean_pose', kwargs.get('mean_pose', None))

        self.val_losses = []

        self.max_grad_norm = 1.0
        self.eps = 1e-8


    def get_input_dim(self):
        return self.input_dim

    def get_text_dim(self):
        return self.text_dim

    def get_pose_dim(self):
        return self.pose_dim


    def diffusion(self, poses, t):
        N, T, C = poses.shape
        noise = torch.randn_like(poses)
        noise_scale = self.noise_level.type_as(noise)[t].unsqueeze(1).unsqueeze(2).repeat(1,T,C)
        noise_scale_sqrt = noise_scale**0.5
        noisy_poses = noise_scale_sqrt * poses + (1.0 - noise_scale)**0.5 * noise
        return noisy_poses, noise

    def denoise(self, noisy_poses, noise_predicted, t=None):

        N, T, C = noisy_poses.shape

        num_noisesteps = self.n_noise_schedule
        if t == None:
            t = torch.randint(0, num_noisesteps, [N], device=noisy_poses.device)
        noise_scale = self.noise_level.type_as(noise_predicted)[t].unsqueeze(1).unsqueeze(2).repeat(1,T,C)
        noise_scale_sqrt = noise_scale**0.5
        pose_predicted = (noisy_poses - (1.0 - noise_scale)**0.5 * noise_predicted) / noise_scale_sqrt
        return pose_predicted


    def forward(self, batch):
        audio_ctrl, clip_cond, poses, memory_motion = batch

        N, T, C = poses.shape

        num_noisesteps = self.n_noise_schedule
        t = torch.randint(0, num_noisesteps, [N], device=poses.device)

        noisy_poses, noise = self.diffusion(poses, t)

        predicted = self.diffusion_model_cross(noisy_poses,
                                               audio_ctrl,
                                               clip_cond,
                                               t,
                                               memory_motion
                                               )


        return noise, predicted, noisy_poses, t


    def synthesize_and_log(self, batch, log_prefix):
        ctrl, g_cond, _, memory_motion = batch
        print("synthesize_and_log")
        clips = self.synthesize(ctrl, g_cond, memory_motion)

        self.log_jerk(clips[:,:,:self.pose_dim], log_prefix)
        file_name = f"{self.current_epoch}_{self.global_step}_{log_prefix}"
        self.log_results(clips.cpu().detach().numpy(), file_name, log_prefix, render_video=True)


    def on_train_start(self):
        if self.memory_parms["use_DMMM"] == True and self.training_stage == 3:
            print("use Mamba Motion Module, freeze other weights")
            for param in self.parameters():
                param.requires_grad = False

            for module in self.diffusion_model_cross.residual_layers:
                if isinstance(module, MultiFeaturesMemoryBlock):
                    for param in module.parameters():
                        param.requires_grad = True
            print("Unfreeze Mamba Motion Module")

        if self.trainer.is_global_zero:
            print("Summarize model")
            summary(self)

        return super().on_train_start()



    def training_step(self, batch, batch_idx):

        if self.training_stage == 3:
            audio_ctrl, clip_cond, poses, memory_motion = batch

            if random.random() < self.ROBUST_TRAINING_NOISE_RATIO:
                clip_cond = clip_cond + torch.randn_like(clip_cond) * self.NOISE_ADD_RATIO
            if random.random() < self.ROBUST_TRAINING_NOISE_RATIO:
                audio_ctrl = audio_ctrl + torch.randn_like(audio_ctrl) * self.NOISE_ADD_RATIO
            if random.random() < self.ROBUST_TRAINING_NOISE_RATIO:
                memory_motion = memory_motion + torch.randn_like(memory_motion) * self.NOISE_ADD_RATIO

            batch = (audio_ctrl, clip_cond, poses, memory_motion)
            noise, predicted, noisy_poses, t = self(batch)

            lossA = self.loss_fn(noise, predicted.squeeze(1))
            # lossM= self.Motion_loss_V2(batch, self.denoise(noisy_poses, predicted.squeeze(1), t))
            self.log('Loss/diffusion', lossA, on_step=True, on_epoch=False, prog_bar=True)
            # self.log('Loss/motion', lossM, on_step=True, on_epoch=False, prog_bar=True)

            loss = lossA # + lossM

        elif self.training_stage == 1 or self.training_stage == 2:
            audio_ctrl, clip_cond, poses = batch

            if self.training_stage == 1:
                if random.random() < self.ROBUST_TRAINING_NOISE_RATIO:
                    clip_cond = clip_cond + torch.randn_like(clip_cond) * self.NOISE_ADD_RATIO
                if random.random() < self.ROBUST_TRAINING_NOISE_RATIO:
                    audio_ctrl = audio_ctrl + torch.randn_like(audio_ctrl) * self.NOISE_ADD_RATIO

            batch = (audio_ctrl, clip_cond, poses, None)
            noise, predicted, noisy_poses, t = self(batch)

            lossA = self.loss_fn(noise, predicted.squeeze(1))
            self.log('Loss/diffusion', lossA, on_step=True, on_epoch=False, prog_bar=True)

            loss = lossA

        self.log('Loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        if self.memory_parms["use_DMMM"] == True:
            pass
        else:
            batch = (*batch, None)

        noise, predicted, _, _ = self(batch)
        loss = self.loss_fn(noise, predicted.squeeze(1))

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        if (self.trainer.global_step > 0 and batch_idx == 0 and
        (self.trainer.current_epoch == self.trainer.max_epochs - 1 or
        self.trainer.current_epoch % self.hparams.Validation["render_every_n_epochs"] == 0) and
        self.trainer.is_global_zero):
            print(f"[{self.trainer.global_rank}] synthesize_and_log start!")
            self.synthesize_and_log(batch, "val")
            print(f"[{self.trainer.global_rank}] synthesize_and_log end!")

        self.val_losses.append(loss)

        output = {"val_loss": loss}
        return output

    def on_validation_epoch_end(self):

        if len(self.val_losses) > 0:
            avg_loss = torch.stack(self.val_losses).mean()
            self.log('Loss/val', avg_loss, sync_dist=True)
            self.val_losses.clear()


    def test_step(self, batch, batch_idx):
        if self.memory_parms["use_DMMM"] == True:
            pass
        else:
            batch = (*batch, None)

        noise, predicted, _ = self(batch)
        loss = self.loss_fn(noise, predicted.squeeze(1))

        self.synthesize_and_log(batch, "test")

        output = {"test_loss": loss}
        return output

    def synthesize(self,
                   ctrl,
                   global_cond,
                   memory_motion=None):
        training_noise_schedule = self.noise_schedule.to(ctrl.device)
        inference_noise_schedule = training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = torch.cumprod(talpha, dim=0)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break

        if len(ctrl.shape) == 2:
            ctrl = ctrl.unsqueeze(0)
            global_cond = global_cond.unsqueeze(0)

        poses = torch.randn(ctrl.shape[0], ctrl.shape[1], self.pose_dim, device=ctrl.device)
        nbatch = poses.size(0)
        noise_scale = (alpha_cum**0.5).type_as(poses).unsqueeze(1)

        for n in tqdm(range(len(alpha) - 1, -1, -1)):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5

            poses = c1 * (poses - c2 * self.diffusion_model_cross(poses, ctrl, global_cond, T[n].unsqueeze(-1),
                                                                  memory_motion).squeeze(1))

            if n > 0:
                noise = torch.randn_like(poses)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                poses += sigma * noise

        anim_clip = self.destandardizeOutput(poses)
        if not self.unconditional:
            out_ctrl = self.destandardizeInput(ctrl)
            anim_clip = torch.cat((anim_clip, out_ctrl), dim=2)

        return anim_clip

    def on_train_end(self):
        return super().on_train_end()

    def check_tensor_health(self, tensor, name="tensor"):
        if torch.isnan(tensor).any():
            print(f"WARNING: NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"WARNING: Inf detected in {name}")
            return False
        if tensor.abs().max() > 1e10:
            print(f"WARNING: Large values detected in {name}: max={tensor.abs().max()}")
            return False
        return True

    def Motion_loss(self, batch, pose_predicted):

        _, _, GT_pose, _ = batch
        deriv = pose_predicted[:, 1:] - pose_predicted[:, :-1]
        acc = deriv[:, 1:] - deriv[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        jerk = jerk.mean(dim=2)

        deriv_GT = GT_pose[:, 1:] - GT_pose[:, :-1]
        acc_GT = deriv_GT[:, 1:] - deriv_GT[:, :-1]
        jerk_GT = acc_GT[:, 1:] - acc_GT[:, :-1]
        jerk_GT = jerk_GT.mean(dim=2)

        return self.loss_motion(jerk, jerk_GT)

    def Motion_loss_V2(self, batch, pose_predicted):
        _, _, GT_pose, _ = batch
        batch_size, seq_len, pose_dim = GT_pose.shape

        if seq_len < 4:
            return torch.tensor(0.0, device=pose_predicted.device, requires_grad=True)

        pose_predicted = torch.clamp(pose_predicted, -1000, 1000)
        GT_pose = torch.clamp(GT_pose, -1000, 1000)

        pred_velocity = pose_predicted[:, 1:] - pose_predicted[:, :-1]
        pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        pred_jerk = pred_acceleration[:, 1:] - pred_acceleration[:, :-1]

        gt_velocity = GT_pose[:, 1:] - GT_pose[:, :-1]
        gt_acceleration = gt_velocity[:, 1:] - gt_velocity[:, :-1]
        gt_jerk = gt_acceleration[:, 1:] - gt_acceleration[:, :-1]

        def simple_normalize(x):
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-6)
            return x / norm, norm.mean()

        pred_velocity_norm, pred_vel_scale = simple_normalize(pred_velocity)
        pred_acceleration_norm, pred_acc_scale = simple_normalize(pred_acceleration)
        pred_jerk_norm, pred_scale = simple_normalize(pred_jerk)

        gt_velocity_norm, gt_vel_scale = simple_normalize(gt_velocity)
        gt_acceleration_norm, gt_acc_scale = simple_normalize(gt_acceleration)
        gt_jerk_norm, gt_scale = simple_normalize(gt_jerk)

        velocity_loss = F.mse_loss(pred_velocity_norm, gt_velocity_norm)
        acceleration_loss = F.mse_loss(pred_acceleration_norm, gt_acceleration_norm)
        jerk_loss = F.mse_loss(pred_jerk_norm, gt_jerk_norm)

        velocity_scale_loss = F.mse_loss(pred_vel_scale, gt_vel_scale)
        acceleration_scale_loss = F.mse_loss(pred_acc_scale, gt_acc_scale)
        jerk_scale_loss = F.mse_loss(pred_scale, gt_scale)

        with torch.no_grad():
            w_pattern = 0.6
            w_scale = 0.4
            w_velocity = 0.3
            w_acceleration = 0.3
            w_jerk = 0.4
            w_global = 0.1

        total_velocity = w_velocity * (w_pattern * velocity_loss + w_scale * velocity_scale_loss)
        total_acceleration = w_acceleration * (w_pattern * acceleration_loss + w_scale * acceleration_scale_loss)
        total_jerk = w_jerk * (w_pattern * jerk_loss + w_scale * jerk_scale_loss)

        motion_loss = w_global * (total_velocity + total_acceleration + total_jerk)

        motion_loss = torch.clamp(motion_loss, max=10.0)

        return motion_loss
