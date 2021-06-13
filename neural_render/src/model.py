# pyright: reportMissingTypeStubs=true

from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import segment_csr
from neural_render_generate_data_full_scene import Constants


class LinearAndMultiply(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_multiply=True,
                 linear_block=nn.Linear):
        super().__init__()

        self._activation = nn.CELU()
        self._linear = linear_block(input_size, output_size)
        self._use_multiply = use_multiply
        if self._use_multiply:
            self._to_multiplier = linear_block(output_size, output_size)

    def forward(self, x, *extra):
        x = self._activation(self._linear(x, *extra))
        if not self._use_multiply:
            return x
        return x * torch.tanh(self._to_multiplier(x, *extra))


def interpolate_sizes(input_size, output_size, count, force_multiple_of=16):
    assert input_size % force_multiple_of == 0
    assert output_size % force_multiple_of == 0
    per = (output_size - input_size) / count
    last_size = input_size
    for i in range(count):
        new_size = round(
            (per *
             (i + 1) + input_size) / force_multiple_of) * force_multiple_of
        yield last_size, new_size
        last_size = new_size

    assert last_size == output_size


def reduce_reshape(x, prefix_sum, dims, reduce="mean"):
    assert len(x.size()) == 2
    assert len(prefix_sum.size()) == 1

    return segment_csr(x, prefix_sum, reduce=reduce).view(*dims, -1)


def values_to_repeated(values, counts):
    return values.view(-1, values.size(-1)).repeat_interleave(counts.view(-1),
                                                              dim=0)


class ResBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_multiply=True,
                 linear_block=nn.Linear,
                 use_norm=True):
        super().__init__()

        self._linear_block = LinearAndMultiply(input_size,
                                               output_size,
                                               use_multiply=False,
                                               linear_block=linear_block)
        self._mul_block = LinearAndMultiply(output_size,
                                            output_size,
                                            use_multiply=use_multiply,
                                            linear_block=linear_block)
        self._use_norm = use_norm
        if self._use_norm:
            self._norm = nn.LayerNorm(output_size)

        self._pad_size = output_size - input_size
        assert self._pad_size >= 0

    def forward(self, x, *extra):
        padded_input = F.pad(x, (0, self._pad_size))
        x = self._mul_block(self._linear_block(x, *extra), *extra)
        x = padded_input + x
        if self._use_norm:
            x = self._norm(x)
        return x


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


class TransformerLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward_multiplier=4,
                 layer_norm_eps=1e-5,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        dim_feedforward = dim_feedforward_multiplier * d_model

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = nn.CELU()

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src.swapaxes(0, 1),
            src.swapaxes(0, 1),
            src.swapaxes(0, 1),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0].swapaxes(0, 1)
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src


class TriangleGeometryBlock(nn.Module):
    def __init__(self):
        super().__init__()

        constants = Constants()

        self._activation = nn.CELU()

        self._tri_size = 256

        self._tri_initial = nn.Linear(constants.n_tri_values, self._tri_size)

        self._n_initial_tri_blocks = 1

        self._initial_tri_blocks = nn.ModuleList([
            ResBlock(self._tri_size, self._tri_size)
            for _ in range(self._n_initial_tri_blocks)
        ])

        self._n_transformer_blocks = 4
        self._n_heads = 8

        self._tri_transformer_blocks = nn.ModuleList([
            TransformerLayer(self._tri_size, self._n_heads)
            for _ in range(self._n_transformer_blocks)
        ])

        self.combined_size = self._tri_size

        self._onto = nn.Linear(self._tri_size, self.combined_size, bias=False)
        self._bsdf = nn.Linear(constants.n_bsdf_values, self.combined_size)
        self._from = nn.Linear(self._tri_size, self.combined_size, bias=True)

        self._n_combined_blocks = 4

        self._combined_blocks = nn.ModuleList([
            ResBlock(self.combined_size, self.combined_size)
            for _ in range(self._n_combined_blocks)
        ])

    def forward(self, inputs):
        # TODO: avoid compute by reshaping where needed
        tri_values = self._activation(
            self._tri_initial(inputs.triangle_features))
        for block in self._initial_tri_blocks:
            tri_values = block(tri_values)

        for block in self._tri_transformer_blocks:
            tri_values = block(tri_values, src_key_padding_mask=inputs.mask)

        # B x S x W -> B x S x S x W
        combined = self._activation(
            self._onto(tri_values).unsqueeze(-2) +
            self._bsdf(inputs.bsdf_features).unsqueeze(-2) +
            self._from(tri_values).unsqueeze(-3))

        for block in self._combined_blocks:
            combined = block(combined)

        return combined


class NextEmissiveBlock(nn.Module):
    def __init__(self, emissive_feat_size, combined_feat_size) -> None:
        super().__init__()
        self._activation = nn.CELU()

        self._to_multiplier = nn.Linear(combined_feat_size, combined_feat_size)
        self._to_combined = nn.Linear(emissive_feat_size, combined_feat_size)

        self._n_blocks = 4

        self._blocks = nn.ModuleList([
            ResBlock(combined_feat_size, combined_feat_size)
            for _ in range(self._n_blocks)
        ])

        self._to_emissive = nn.Linear(combined_feat_size, emissive_feat_size)

    def forward(self, emissions, combined, mask):
        multiplier = torch.sigmoid(self._to_multiplier(combined))
        overall = self._to_combined(
            emissions.unsqueeze(-3)) * multiplier + combined

        for block in self._blocks:
            overall = block(overall)

        # zero diagonal
        assert overall.size(-2) == overall.size(-3)
        torch.diagonal(overall, dim1=-3, dim2=-2)[:] = 0

        overall[mask] = 0

        # sum over the axis along which emissions and _from vary (axis after
        # unsqueeze)
        return self._to_emissive(overall.sum(axis=-2))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self._activation = nn.CELU()
        constants = Constants()

        self._tri_initial = TriangleGeometryBlock()

        self._emission_feature_size = 512
        self._emission_to_initial_feat = nn.Linear(constants.n_rgb_dims,
                                                   self._emission_feature_size,
                                                   bias=False)

        self._emission_block = NextEmissiveBlock(
            self._emission_feature_size, self._tri_initial.combined_size)

        self._coord_size = constants.n_coords_feature_values
        self._coord_block_size = 32
        self._coords_expand = nn.Linear(self._coord_size,
                                        self._coord_block_size)
        self._coords_block = ResBlock(self._coord_block_size,
                                      self._coord_block_size)
        self._coords_to_multiplier = nn.Linear(self._coord_block_size,
                                               self._emission_feature_size)

        self._output_block_size = 32

        self._coords_to_addr = nn.Linear(self._coord_block_size,
                                         self._output_block_size)

        self._to_output_block = nn.Linear(self._emission_feature_size,
                                          self._output_block_size)
        self._output_block = ResBlock(self._output_block_size,
                                      self._output_block_size)

        self._output_size = 3

        self._output = nn.Linear(self._output_block_size, self._output_size)

    def forward(self, inputs, steps):
        values = self._tri_initial(inputs)

        y = self._coords_block(
            self._activation(self._coords_expand(inputs.baryocentric_coords)))
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))

        all_lighting = []
        emissions = self._emission_to_initial_feat(inputs.emissive_values)

        square_mask = torch.logical_or(inputs.mask.unsqueeze(-1),
                                       inputs.mask.unsqueeze(-2))

        # skip first step (which is just direct light)
        for _ in range(steps - 1):
            emissions = self._emission_block(emissions, values, square_mask)

            assert emissions.size(0) == inputs.triangle_idxs_for_coords.size(0)
            out = self._to_output_block(
                emissions.view(-1,
                               *emissions.size()[2:])
                [inputs.triangle_idxs_for_coords.view(-1)].view(
                    emissions.size(0), inputs.triangle_idxs_for_coords.size(1),
                    *emissions.size()[2:]) * multiplier)

            lighting = torch.relu(self._output(self._output_block(out)))

            # zero out other values (convention for loss)
            # this current approach is not super clean or efficient...
            assert len(inputs.n_samples_per) == lighting.size(0)
            for i, count in enumerate(inputs.n_samples_per):
                lighting[i, count:] = 0

            all_lighting.append(lighting)

        return torch.stack(all_lighting, dim=1)


if __name__ == "__main__":
    assert len(list(interpolate_sizes(128, 496, 8))) == 8
    assert len(list(interpolate_sizes(496, 496, 8))) == 8
    assert len(list(interpolate_sizes(1024, 496, 8))) == 8
    for inp_s, out_s in interpolate_sizes(128, 496, 8):
        print("inp:", inp_s, "out:", out_s)
