import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from mamba_ssm.modules.mamba2_simple import Mamba2Simple

from mamba_ssm import Mamba
from mamba_ssm import Mamba2
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block,_init_weights

class Conv1dLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
    super().__init__()
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(self.conv1d.weight)

  def forward(self, x):
    return self.conv1d(x.permute(0,2,1)).permute(0,2,1)

def silu(x):
  return x * torch.sigmoid(x)


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 ):
        super().__init__()
        self.mamba_block = Mamba(
                            d_model,
                            d_state,
                            d_conv=d_conv,
                            expand=expand,
                            dt_rank=dt_rank,
                            dt_min=dt_min,
                            dt_max=dt_max,
                            dt_init=dt_init,
                            dt_scale=dt_scale,
                            dt_init_floor=dt_init_floor)
    def forward(self, x):
        x = self.mamba_block(x)
        return x


class Mamba2Simple(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,

                 d_model,
                 num_blocks,
                 d_state,
                 d_conv,
                 expand,

                 conv_init,
                 headdim,
                 D_has_hdim,
                 rmsnorm,
                 norm_before_gate,
                 dt_min,
                 dt_max,
                 dt_init_floor,
                 bias,
                 conv_bias
                 ):
        super(Mamba2Simple,self).__init__()
        self.in_proj = nn.Linear(in_channels, d_model)

        self.mamba_blocks = nn.ModuleList(
            [
                Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, conv_init=conv_init, headdim=headdim, D_has_hdim=D_has_hdim, rmsnorm=rmsnorm, norm_before_gate=norm_before_gate, dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor, bias=bias, conv_bias=conv_bias)
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.mamba_blocks:
            x = layer(x)
        x = self.out_proj(x)
        return x


class DualMemoryMamba(nn.Module):
    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand=2):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.mask_proj = nn.Linear(1, dim)

    def forward(self, hidden_state, mask=None):
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            mask_features = self.mask_proj(mask_expanded)

            x_masked = hidden_state * mask_expanded + mask_features * (1 - mask_expanded)
        else:
            x_masked = hidden_state

        forward_out = self.forward_mamba(x_masked)

        x_rev = torch.flip(x_masked, [1])
        backward_out = self.backward_mamba(x_rev)
        backward_out = torch.flip(backward_out, [1])

        ssm_output = forward_out + backward_out

        if mask is not None:
            output = ssm_output * mask_expanded + hidden_state * (1 - mask_expanded)
        else:
            output = ssm_output

        return output

class ModalityBranch(nn.Module):
    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand=2):
        super().__init__()
        self.mask_ssm = DualMemoryMamba(dim, d_state, d_conv, expand)

    def forward(self, hidden_state,
                motion_embedding=None, motion_mask=None):

        batch_size, seq_len, _ = hidden_state.shape

        if motion_embedding is not None:
            _, memory_len, _ = motion_embedding.shape
            hidden_state = torch.cat([motion_embedding, hidden_state], dim=1)

        output = self.mask_ssm(hidden_state, motion_mask)

        if motion_embedding is not None:
            _, output= torch.split(output, [memory_len, seq_len], dim=1)

        return output

class MultiFeaturesMemoryBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,

                 motion_dim,
                 memory_args=None
                 ):
        super().__init__()


        if memory_args["activation"]=="relu":
            self.act = nn.ReLU()
        elif memory_args["activation"]=="gelu":
            self.act = nn.GELU()
        elif memory_args["activation"]=="elu":
            self.act = nn.ELU()

        self.f_theta1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
        )

        self.motion_projection = Conv1dLayer(motion_dim, hidden_dim, 1)


        self.memory_branch = ModalityBranch(hidden_dim,
                                            memory_args['d_state'],
                                            memory_args['d_conv'],
                                            memory_args['expand'])


        self.norm = nn.LayerNorm(hidden_dim)

        self.f_theta2 = nn.Sequential(
            self.act,
            nn.Linear(hidden_dim // 2, 2 * input_dim)
        )

    def forward(self,
                x,
                motion_memory_emb,
                G1: bool=True,
                motion_mask=None):
        hidden_state = self.f_theta1(x)

        motion_memory_emb = self.motion_projection(motion_memory_emb)

        memory_out = self.memory_branch(hidden_state, motion_memory_emb, motion_mask) if G1 else hidden_state

        fused_features = memory_out

        fused_features = self.norm(fused_features)

        output = fused_features + hidden_state

        gate, filter = torch.chunk(output, 2, dim=2)

        output = torch.sigmoid(gate) * torch.tanh(filter)

        output = self.f_theta2(output)
        residual, skip = torch.chunk(output, 2, dim=2)
        return (x + residual) / sqrt(2.0), skip
