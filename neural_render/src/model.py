import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LinearAndMultiply(nn.Module):
    def __init__(self, input_size, output_size, use_multiply=True):
        super().__init__()

        self._activation = nn.CELU()
        self._linear = nn.Linear(input_size, output_size)
        self._use_multiply = use_multiply
        if self._use_multiply:
            self._to_multiplier = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self._activation(self._linear(x))
        if not self._use_multiply:
            return x
        return x * torch.tanh(self._to_multiplier(x))


def interpolate_sizes(input_size, output_size, count):
    per = (output_size - input_size) / count
    last_size = input_size
    for i in range(count):
        new_size = round(per * (i + 1) + input_size)
        yield last_size, new_size
        last_size = new_size

    assert last_size == output_size


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))

    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)

    return x.view(*s[:-n_dims], -1)


class CoordAttention(nn.Module):
    """ Multi-Headed Dot Product Attention for coord values """
    def __init__(self,
                 value_input_size,
                 coord_input_size,
                 key_size,
                 output_size,
                 n_heads,
                 overall_weighting=True,
                 mix_bias=0.0):
        super().__init__()

        self.overall_weighting = overall_weighting
        self.output_size = output_size
        self.mix_bias = mix_bias

        self.n_heads = n_heads

        self._proj_k = nn.Linear(value_input_size, key_size)
        self._proj_v = nn.Linear(value_input_size, output_size * self.n_heads)
        self._proj_q = nn.Linear(coord_input_size, key_size)

        if self.overall_weighting:
            self._overall_gain = nn.Parameter(torch.Tensor(1))
            self._overall_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.overall_weighting:
            nn.init.ones_(self._overall_gain)
            nn.init.constant_(self._overall_bias, self.mix_bias)

    def forward(self, pre_value_key, coords):
        """
        pre_value_key is used to compute value/key
        coords is used to compute query
        pre_value_key has size (..., D_v)
        coords has size (..., D_q)

        key size is D_k
        output size is D_o
        """

        # (..., D_q) -proj-> (..., D_k)
        q = self._proj_q(coords)
        # (..., D_v) -proj-> (..., D_k)
        k = self._proj_k(pre_value_key)
        # (..., D_v) -proj-> (..., D_o * H)
        v = self._proj_v(pre_value_key)

        # (..., D) -split-> (..., H, W)
        q, k, v = (split_last(x, (self.n_heads, -1)) for x in [q, k, v])

        # (..., H, W) * (..., H, W) -dot-> (..., H)
        scores = (q * k).sum(axis=-1)

        overall_weight = None
        if self.overall_weighting:
            # (..., H) -mean-> (..., 1)
            pooled_scores = scores.mean(-1, keepdim=True)
            overall_weight = torch.sigmoid(pooled_scores * self._overall_gain +
                                           self._overall_bias)

        # (..., H) -softmax-> (..., H)
        softmax_scores = F.softmax(scores, dim=-1)

        # (..., H, 1) * (..., H, D_o) -sum-> (..., D_o)
        h = (softmax_scores.unsqueeze(-1) * v).sum(-2)

        if self.overall_weighting:
            out = h * overall_weight
        else:
            out = h

        return out, overall_weight


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, use_multiply=True):
        super().__init__()

        self._linear_block = LinearAndMultiply(input_size,
                                               output_size,
                                               use_multiply=False)
        self._mul_block = LinearAndMultiply(output_size,
                                            output_size,
                                            use_multiply=use_multiply)
        self._norm = nn.LayerNorm(output_size)

        self._pad_size = output_size - input_size
        assert self._pad_size >= 0

    def forward(self, x):
        padded_input = F.pad(x, (0, self._pad_size))
        x = self._mul_block(self._linear_block(x))
        return self._norm(padded_input + x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self._activation = nn.CELU()
        self._input_size = 37
        self._start_size = 256
        self._end_size = 256
        self._multiplier_size = 256

        self._input_expand = nn.Linear(self._input_size, self._start_size)
        self._final = nn.Linear(self._end_size, self._multiplier_size)

        self._n_blocks = 8

        scene_sizes = interpolate_sizes(self._start_size, self._end_size,
                                        self._n_blocks)
        self._scene_blocks = nn.ModuleList(
            [ResBlock(inp, out) for inp, out in scene_sizes])

        self._coord_size = 2
        self._coord_block_size = 32
        self._coords_expand = nn.Linear(self._coord_size,
                                        self._coord_block_size)
        self._coords_block = ResBlock(self._coord_block_size,
                                      self._coord_block_size)
        self._coords_to_multiplier = nn.Linear(self._coord_block_size,
                                               self._multiplier_size)

        self._output_block_size = 64

        self._to_output_block = nn.Linear(self._multiplier_size,
                                          self._output_block_size)

        self._values_per_head = 8
        self._n_heads = 128
        self._use_attn_weight = True

        self._attn = CoordAttention(self._end_size,
                                    self._coord_block_size,
                                    self._values_per_head * self._n_heads,
                                    self._output_block_size,
                                    self._n_heads,
                                    overall_weighting=self._use_attn_weight,
                                    mix_bias=0)

        self._output_block = ResBlock(self._output_block_size,
                                      self._output_block_size)

        self._output_size = 3

        self._output = nn.Linear(self._output_block_size, self._output_size)

    def forward(self, scenes, coords):
        assert scenes.size(0) == coords.size(0)

        x = self._activation(self._input_expand(scenes))
        for block in self._scene_blocks:
            x = block(x)

        y = self._coords_block(self._activation(self._coords_expand(coords)))
        x = x.unsqueeze(1)

        assert len(y.size()) == len(x.size())

        # TODO are there redundant params/should there be activation
        # here (or later before out is put into block)
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))
        multiplied = self._final(x) * multiplier

        attn, attn_weight = self._attn(x, y)

        if self._use_attn_weight:
            # attn_weight is already multiplied by _attn
            out = (1 - attn_weight) * multiplied + attn
        else:
            out = multiplied + attn

        out = self._to_output_block(out)

        return torch.relu(self._output(self._output_block(out)))


if __name__ == "__main__":
    assert len(list(interpolate_sizes(128, 507, 8))) == 8
    assert len(list(interpolate_sizes(507, 507, 8))) == 8
    for inp_s, out_s in interpolate_sizes(128, 507, 8):
        print("inp:", inp_s, "out:", out_s)
