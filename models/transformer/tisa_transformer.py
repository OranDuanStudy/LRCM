import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.tisa_v2 import TisaV2


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)


class TisaTransformer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        d_model,
        num_blocks,
        num_heads,
        activation,
        norm,
        drop_prob,
        d_ff=2048,
        tisa_num_kernels=21,
        seqlen=128,
        use_preln=False,
        bias=False,
        dilation=1
    ):
        super(TisaTransformer, self).__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.attention_blocks = nn.ModuleList(
            [
                AttnBlock(d_model, num_heads, activation, norm, drop_prob, tisa_num_kernels, seqlen, use_preln, d_ff, bias=bias, dilation=dilation)
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.attention_blocks:
            x = layer(x)
        x = self.out_proj(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class AttnBlock(nn.Module):
    def __init__(self, d_model, num_heads, activation, norm, drop_prob, tisa_num_kernels, seqlen, use_preln=False, d_ff=2048, bias=False, dilation=1):
        super(AttnBlock, self).__init__()
        self.use_preln = use_preln
        self.attn = GatedAttn(d_model, num_heads=num_heads, activation=activation, seqlen=seqlen, drop_prob=drop_prob, tisa_num_kernels=tisa_num_kernels)
        if (dilation>0):
            self.ff = ConvLayer(d_model, d_ff, activation=activation, dilation=dilation, dropout=drop_prob, bias=bias)
        else:
            self.ff = FeedForward(d_model, d_ff, activation=activation, dropout=drop_prob, bias=bias)

        if norm == "T5":
            self.norm_1 = T5LayerNorm(d_model)
            self.norm_2 = T5LayerNorm(d_model)
        elif norm == "LN":
            self.norm_1 = nn.LayerNorm(d_model)
            self.norm_2 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"unknown norm: {norm}")

        self.dropout_1 = nn.Dropout(drop_prob)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, x):
        if self.use_preln:
            x = self.dropout_1(self.attn(self.norm_2(x))) + x
        else:
            x = self.dropout_1(self.attn(x)) + x
            x = self.norm_2(x)

        if self.use_preln:
            x = self.dropout_2(self.ff(self.norm_1(x))) + x
        else:
            x = self.dropout_2(self.ff(x)) + x
            x = self.norm_1(x)

        return x

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states

class ConvLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, activation="relu", dilation=1, dropout = 0.1, bias=False):
        super().__init__()
        self.conv_1 = nn.Conv1d(d_model, d_ff, 3, padding=dilation, dilation=dilation, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(d_ff, d_model, 3, padding=dilation, dilation=dilation, bias=bias)
        if activation=="relu":
            self.act = nn.ReLU()
        elif activation=="gelu":
            self.act = nn.GELU()
        elif activation=="elu":
            self.act = nn.ELU()

    def forward(self, x):
        x = self.dropout(self.act(self.conv_1(x.permute(0,2,1))))
        x = self.conv_2(x)
        return x.permute(0,2,1)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, activation="RELU", dropout = 0.1, bias=False):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=bias)
        if activation=="relu":
            self.act = nn.ReLU()
        elif activation=="gelu":
            self.act = nn.GELU()
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class GatedAttn(nn.Module):
    """Gated Multi-Head Self-Attention Block

    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, d_model, num_heads=4, activation="RELU", seqlen=128, drop_prob=0.0, tisa_num_kernels=21):
        super(GatedAttn, self).__init__()

        assert d_model % num_heads==0, f"num_heads ({num_heads}) is not evenly divisible by d_model ({d_model})"

        self.d_model = d_model
        self.num_heads = num_heads
        if drop_prob>0:
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.dropout = None

        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate = nn.Linear(d_model, 2 * d_model)
        self.key_depth_per_head = torch.tensor(self.d_model / self.num_heads)

        self.position_scorer = TisaV2(self.num_heads,
                                        tisa_num_kernels=tisa_num_kernels,
                                        tisa_dropout_prob=drop_prob,
                                        num_position_agnostic_heads=0,
                                        max_field_view=seqlen//2,
                                        min_field_view=5)

        self.position_scorer._init_weights()


    def forward(self, x):
        _, seq_len, num_channels = x.size()

        memory, query = torch.split(self.in_proj(x), (2 * num_channels, num_channels), dim=-1)


        q = self.split_last_dim(query, self.num_heads)
        k, v = [
            self.split_last_dim(tensor, self.num_heads)
            for tensor in torch.split(memory, self.d_model, dim=2)
        ]

        x = self.dot_product_attention(q, k, v)

        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))


        x = self.gate(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v):
        """Dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights /= torch.sqrt(self.key_depth_per_head)

        seq_len = weights.shape[-1]

        weights += self.position_scorer(seq_len)

        weights = F.softmax(weights, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)

        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        if last is not None:
            new_shape = old_shape[:-1] + [n] + [last // n]
        else:
            new_shape = old_shape[:-1] + [n]

        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b]
        ret = x.contiguous().view(new_shape)

        return ret



class CrossAttn(nn.Module):
    """Gated CrossAttn Multi-Head Cross-Attention Block

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, d_model, num_heads=4, activation="RELU", seqlen=128, drop_prob=0.0, tisa_num_kernels=21):
        super().__init__()

        assert d_model % num_heads == 0, f"num_heads ({num_heads}) is not evenly divisible by d_model ({d_model})"

        self.d_model = d_model
        self.num_heads = num_heads
        if drop_prob > 0:
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.dropout = None

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.memory_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        self.gate = nn.Linear(d_model, 2 * d_model)
        self.key_depth_per_head = torch.tensor(self.d_model / self.num_heads)

        self.position_scorer = TisaV2(self.num_heads,
                                        tisa_num_kernels=tisa_num_kernels,
                                        tisa_dropout_prob=drop_prob,
                                        num_position_agnostic_heads=0,
                                        max_field_view=seqlen//2,
                                        min_field_view=5)

        self.position_scorer._init_weights()


    def forward(self, x_q, x_kv):
        b_q, seq_len_q, d_q = x_q.size()
        b_kv, seq_len_kv, d_kv = x_kv.size()

        query = self.query_proj(x_q)
        memory = self.memory_proj(x_kv)


        q = self.split_last_dim(query, self.num_heads)
        k, v = [
            self.split_last_dim(tensor, self.num_heads)
            for tensor in torch.split(memory, self.d_model, dim=2)
        ]

        x = self.dot_product_attention(q, k, v)

        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))

        x = self.gate(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v):
        """Dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights /= torch.sqrt(self.key_depth_per_head)

        seq_len = weights.shape[-1]

        weights += self.position_scorer(seq_len)

        weights = F.softmax(weights, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)

        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        if last is not None:
            new_shape = old_shape[:-1] + [n] + [last // n]
        else:
            new_shape = old_shape[:-1] + [n]

        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b]
        ret = x.contiguous().view(new_shape)

        return ret


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads, activation, norm, drop_prob, tisa_num_kernels, seqlen, use_preln=False, d_ff=2048, bias=False, dilation=1):
        super().__init__()
        self.use_preln = use_preln
        self.attn = CrossAttn(d_model, num_heads=num_heads, activation=activation, seqlen=seqlen, drop_prob=drop_prob, tisa_num_kernels=tisa_num_kernels)

        if (dilation>0):
            self.ff = ConvLayer(d_model, d_ff, activation=activation, dilation=dilation, dropout=drop_prob, bias=bias)
        else:
            self.ff = FeedForward(d_model, d_ff, activation=activation, dropout=drop_prob, bias=bias)

        if norm == "T5":
            self.norm_x1 = T5LayerNorm(d_model)
            self.norm_x2 = T5LayerNorm(d_model)

            self.norm_y = T5LayerNorm(d_model)
        elif norm == "LN":
            self.norm_x1 = nn.LayerNorm(d_model)
            self.norm_x2 = nn.LayerNorm(d_model)

            self.norm_y = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"unknown norm: {norm}")

        self.dropout_1 = nn.Dropout(drop_prob)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, x, y):
        if self.use_preln:
            x = self.dropout_1(
                self.attn(self.norm_x1(x) , self.norm_y(y))) + x
        else:
            x = self.dropout_1(
                self.attn(x , self.norm_y(y))) + x
            x = self.norm_x1(x)

        if self.use_preln:
            x = self.dropout_2(self.ff(
                self.norm_x2(x))) + x
        else:
            x = self.dropout_2(self.ff(x)) + x
            x = self.norm_x2(x)

        return x

class CrossTransformer(nn.Module):

    def __init__(
        self,
        in_channels,
        clip_channels,

        out_channels,
        d_model,
        num_blocks,
        num_heads,
        activation,
        norm,
        drop_prob,
        d_ff=2048,
        tisa_num_kernels=21,
        seqlen=128,
        use_preln=False,
        bias=False,
        dilation=1
    ):
        super().__init__()
        self.in_proj_q = nn.Linear(in_channels, d_model)
        self.in_proj_kv = nn.Linear(clip_channels, d_model)

        self.attention_blocks = nn.ModuleList(
            [
                CrossAttnBlock(d_model, num_heads, activation, norm, drop_prob, tisa_num_kernels, seqlen, use_preln, d_ff, bias=bias, dilation=dilation)
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, x , y):
        x = self.in_proj_q(x)
        y = self.in_proj_kv(y)

        for layer in self.attention_blocks:
            x = layer(x , y)
        x = self.out_proj(x)
        return x


class DualTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        clip_channels,
        out_channels,
        d_model,
        num_blocks,
        num_heads,
        activation,
        norm,
        drop_prob,
        d_ff=2048,
        tisa_num_kernels=21,
        seqlen=128,
        use_preln=False,
        bias=False,
        dilation=1
    ):
        super().__init__()

        self.tisa_in_proj = nn.Linear(in_channels, d_model)
        self.cross_in_proj_kv = nn.Linear(clip_channels, d_model)
        self.out_proj = nn.Linear(d_model, out_channels)

        self.tisa_blocks = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.tisa_blocks.append(
                AttnBlock(
                    d_model, num_heads, activation, norm, drop_prob,
                    tisa_num_kernels, seqlen, use_preln, d_ff,
                    bias=bias, dilation=dilation
                )
            )
            self.cross_blocks.append(
                CrossAttnBlock(
                    d_model, num_heads, activation, norm, drop_prob,
                    tisa_num_kernels, seqlen, use_preln, d_ff,
                    bias=bias, dilation=dilation
                )
            )

    def forward(self, x, y):
        x = self.tisa_in_proj(x)
        y = self.cross_in_proj_kv(y)

        for tisa_block, cross_block in zip(self.tisa_blocks, self.cross_blocks):
            x = tisa_block(x)
            x = cross_block(x, y)

        x = self.out_proj(x)
        return x
