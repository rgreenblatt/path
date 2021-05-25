from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from neural_render_generate_data import Constants
from torch_scatter import segment_csr


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


class PolyConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # only 1 bias needed
        self._layer = nn.Linear(input_size * 3, output_size)

    def forward(self, x, item_to_left_idxs, item_to_right_idxs):
        assert len(x.size()) == 2
        assert len(item_to_left_idxs.size()) == 1
        assert len(item_to_right_idxs.size()) == 1
        return self._layer(
            torch.cat((x, x[item_to_left_idxs], x[item_to_right_idxs]),
                      dim=-1))


def poly_reduce_reshape(x, prefix_sum, dims):
    assert len(x.size()) == 2
    assert len(prefix_sum.size()) == 1

    return segment_csr(x, prefix_sum, reduce="mean").view(*dims, -1)


def values_to_poly_points(values, counts):
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


PolyBlock = partial(ResBlock, linear_block=PolyConv)


class FusedBlock(nn.Module):
    def __init__(self, inp_overall, out_overall, inp_poly, out_poly):
        super().__init__()
        constants = Constants()
        self._activation = nn.CELU()
        self._inp_poly = inp_poly
        self._overall_to_poly = nn.Linear(inp_overall,
                                          inp_poly * constants.n_polys)
        # norm is covered by '_overall_norm'
        self._overall_block = ResBlock(inp_overall,
                                       out_overall,
                                       use_norm=False)

        self._poly_layer_norms = nn.ModuleList(
            [nn.LayerNorm(inp_poly) for _ in range(constants.n_polys)])
        self._poly_blocks = nn.ModuleList(
            [PolyBlock(inp_poly, out_poly) for _ in range(constants.n_polys)])
        self._poly_to_overall = nn.ModuleList([
            nn.Linear(out_poly, out_overall) for _ in range(constants.n_polys)
        ])
        self._poly_empty_values_for_overall = nn.ParameterList(
            [nn.Parameter(torch.Tensor(out_overall))])

        self.reset_parameters()

        self._overall_norm = nn.LayerNorm(out_overall)

    def reset_parameters(self):
        for param in self._poly_empty_values_for_overall:
            nn.init.zeros_(param)

    def forward(self, overall_values, poly_values):
        overall_for_poly = self._activation(
            self._overall_to_poly(overall_values))

        overall_values = self._overall_block(overall_values)

        new_poly_values = []
        for i, (layer_norm, block, to_overall, empty_values,
                (features, poly_value)) in enumerate(
                    zip(self._poly_layer_norms, self._poly_blocks,
                        self._poly_to_overall,
                        self._poly_empty_values_for_overall, poly_values)):
            poly_value = poly_value + values_to_poly_points(
                overall_for_poly[..., i * self._inp_poly:(i + 1) *
                                 self._inp_poly], features.counts)
            poly_value = layer_norm(poly_value)
            poly_value = block(poly_value, features.item_to_left_idxs,
                               features.item_to_right_idxs)

            reduced = poly_reduce_reshape(poly_value,
                                          features.prefix_sum_counts,
                                          features.counts.size())
            base_overall = self._activation(to_overall(reduced))
            base_overall[features.counts == 0] = empty_values

            overall_values = overall_values + base_overall

            new_poly_values.append((features, poly_value))

        overall_values = self._overall_norm(overall_values)

        return overall_values, new_poly_values


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self._activation = nn.CELU()
        constants = Constants()

        self._start_tri_size = 64
        self._n_initial_tri_blocks = 1

        self._tri_initial = nn.Linear(constants.n_tri_values,
                                      self._start_tri_size)
        self._initial_tri_blocks = nn.ModuleList([
            ResBlock(self._start_tri_size, self._start_tri_size)
            for _ in range(self._n_initial_tri_blocks)
        ])

        self._start_poly_size = 64
        self._tri_initial_for_poly = nn.Linear(self._start_tri_size,
                                               self._start_poly_size,
                                               bias=True)

        self._poly_feature_initial = nn.Linear(constants.n_poly_feature_values,
                                               self._start_poly_size,
                                               bias=False)

        self._poly_point_initial = nn.Linear(constants.n_poly_point_values,
                                             self._start_poly_size,
                                             bias=False)

        self._n_shared_poly_block = 1
        self._shared_poly_blocks = nn.ModuleList([
            PolyBlock(self._start_poly_size, self._start_poly_size)
            for _ in range(self._n_shared_poly_block)
        ])

        self._initial_overall_size = 256

        self._overall_initial = nn.Linear(
            constants.n_scene_values + self._start_tri_size * 3,
            self._initial_overall_size)

        self._n_initial_overall_blocks = 1
        self._initial_overall_blocks = nn.ModuleList([
            ResBlock(self._initial_overall_size, self._initial_overall_size)
            for _ in range(self._n_initial_overall_blocks)
        ])

        self._n_fused_blocks = 4
        self._end_fused_size = 256
        self._end_poly_size = self._start_poly_size

        overall_sizes = interpolate_sizes(self._initial_overall_size,
                                          self._end_fused_size,
                                          self._n_fused_blocks)
        poly_sizes = interpolate_sizes(self._start_poly_size,
                                       self._end_poly_size,
                                       self._n_fused_blocks)
        self._fused_blocks = nn.ModuleList([
            FusedBlock(inp_overall, out_overall, inp_poly, out_poly)
            for ((inp_overall, out_overall),
                 (inp_poly, out_poly)) in zip(overall_sizes, poly_sizes)
        ])

        self._end_size = self._end_fused_size

        self._n_final_overall_blocks = 4
        overall_sizes = interpolate_sizes(self._end_fused_size, self._end_size,
                                          self._n_final_overall_blocks)
        self._final_overall_blocks = nn.ModuleList(
            [ResBlock(inp, out) for inp, out in overall_sizes])

        self._multiplier_size = self._end_size

        self._final = nn.Linear(self._end_size, self._multiplier_size)

        self._coord_size = constants.n_coords_feature_values
        self._coord_block_size = 32
        self._coords_expand = nn.Linear(self._coord_size,
                                        self._coord_block_size)
        self._coords_block = ResBlock(self._coord_block_size,
                                      self._coord_block_size)
        self._coords_to_multiplier = nn.Linear(self._coord_block_size,
                                               self._multiplier_size)

        self._output_block_size = 64

        self._coords_to_addr = nn.Linear(self._coord_block_size,
                                         self._output_block_size)

        self._to_output_block = nn.Linear(self._multiplier_size,
                                          self._output_block_size)
        self._output_block = ResBlock(self._output_block_size,
                                      self._output_block_size)

        self._output_size = 3

        self._output = nn.Linear(self._output_block_size, self._output_size)

    def forward(self, inputs):
        inputs = inputs.item
        tri_values = self._activation(
            self._tri_initial(inputs.triangle_features))
        for block in self._initial_tri_blocks:
            tri_values = block(tri_values)

        tri_for_poly = self._tri_initial_for_poly(tri_values)
        poly_values = []
        for poly_input in inputs.polygon_inputs:
            features = poly_input.polygon_feature
            tri_for_this_poly = tri_for_poly[:, poly_input.tri_idx]

            tri_for_this_poly = values_to_poly_points(tri_for_this_poly,
                                                      features.counts)

            overall = self._poly_feature_initial(features.overall_features)
            overall = values_to_poly_points(overall, features.counts)

            poly_point_vals = self._poly_point_initial(features.point_values)

            x = self._activation(tri_for_this_poly + overall + poly_point_vals)
            for block in self._shared_poly_blocks:
                x = block(x, features.item_to_left_idxs,
                          features.item_to_right_idxs)
            poly_values.append((features, x))

        overall_values = self._activation(
            self._overall_initial(
                torch.cat((inputs.overall_scene_features,
                           tri_values.view(tri_values.size(0), -1)),
                          dim=-1)))

        for block in self._initial_overall_blocks:
            overall_values = block(overall_values)

        for block in self._fused_blocks:
            (overall_values, poly_values) = block(overall_values, poly_values)

        # done with poly
        x = overall_values

        for block in self._final_overall_blocks:
            x = block(x)
        x = self._activation(self._final(x))

        y = self._coords_block(
            self._activation(self._coords_expand(inputs.baryocentric_coords)))
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))
        out = self._to_output_block(torch.unsqueeze(x, 1) * multiplier)

        return torch.relu(self._output(self._output_block(out)))


if __name__ == "__main__":
    assert len(list(interpolate_sizes(128, 496, 8))) == 8
    assert len(list(interpolate_sizes(496, 496, 8))) == 8
    assert len(list(interpolate_sizes(1024, 496, 8))) == 8
    for inp_s, out_s in interpolate_sizes(128, 496, 8):
        print("inp:", inp_s, "out:", out_s)
