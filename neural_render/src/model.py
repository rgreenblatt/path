from functools import partial

import numpy as np
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


def poly_reduce_reshape(x, prefix_sum, dims, reduce="mean"):
    assert len(x.size()) == 2
    assert len(prefix_sum.size()) == 1

    return segment_csr(x, prefix_sum, reduce=reduce).view(*dims, -1)


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


PolyBlock = partial(ResBlock, linear_block=PolyConv)


class FusedBlock(nn.Module):
    def __init__(self, inp_overall, out_overall, inp_poly, out_poly, inp_ray,
                 out_ray):
        super().__init__()
        constants = Constants()
        self._activation = nn.CELU()
        self._inp_poly = inp_poly
        self._inp_ray = inp_ray
        self._overall_to_poly = nn.Linear(inp_overall,
                                          inp_poly * constants.n_polys)
        self._overall_to_ray = nn.Linear(inp_overall,
                                         inp_ray * constants.n_ray_items)
        # norm is covered by '_overall_norm'
        self._overall_block = ResBlock(inp_overall,
                                       out_overall,
                                       use_norm=False)

        self._n_heads_for_ray = 8
        n_feat_per_head = 16
        self._key_size = self._n_heads_for_ray * n_feat_per_head

        self._ray_layer_norms = nn.ModuleList(
            [nn.LayerNorm(inp_ray) for _ in range(constants.n_ray_items)])
        self._ray_blocks = nn.ModuleList(
            [ResBlock(inp_ray, out_ray) for _ in range(constants.n_ray_items)])
        self._ray_to_overall = nn.ModuleList([
            nn.Linear(out_ray, out_overall)
            for _ in range(constants.n_ray_items)
        ])
        self._ray_to_key = nn.ModuleList([
            nn.Linear(out_ray, self._key_size)
            for _ in range(constants.n_ray_items)
        ])
        self._overall_to_query = nn.Linear(
            out_overall, self._key_size * constants.n_ray_items)

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

    def forward(self, overall_values, poly_values, ray_values):
        # could also put these after block (done this way to allow async
        # execution)
        overall_for_poly = self._activation(
            self._overall_to_poly(overall_values))

        overall_for_ray = self._activation(
            self._overall_to_ray(overall_values))

        overall_values = self._overall_block(overall_values)

        query_for_ray = self._overall_to_query(overall_values)

        new_ray_values = []
        for i, (layer_norm, block, to_overall, to_key,
                (features, ray_values)) in enumerate(
                    zip(self._ray_layer_norms, self._ray_blocks,
                        self._ray_to_overall, self._ray_to_key, ray_values)):
            ray_values = ray_values + values_to_poly_points(
                overall_for_ray[..., i * self._inp_ray:(i + 1) *
                                self._inp_ray], features.counts)
            ray_values = layer_norm(ray_values)
            ray_values = block(ray_values)

            new_ray_values.append((features, ray_values))

            # attention

            for_overall = to_overall(ray_values)
            key = to_key(ray_values)
            query = values_to_poly_points(
                query_for_ray[...,
                              i * self._key_size:(i + 1) * self._key_size],
                features.counts)
            [for_overall, key, query] = [
                split_last(x, (self._n_heads_for_ray, -1))
                for x in (for_overall, key, query)
            ]
            weights = torch.exp((query * key).sum(axis=-1))
            # divide by total for softmax
            total_weights = poly_reduce_reshape(weights,
                                                features.prefix_sum_counts,
                                                features.counts.size(),
                                                reduce='sum')
            weights = weights / values_to_poly_points(
                total_weights,
                features.counts,
            )

            overall_values = overall_values + poly_reduce_reshape(
                merge_last(weights.unsqueeze(-1) * for_overall, 2),
                features.prefix_sum_counts,
                features.counts.size(),
                reduce='sum')

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
            # TODO: consider precomputing the '== 0'
            base_overall[features.counts == 0] = empty_values

            overall_values = overall_values + base_overall

            new_poly_values.append((features, poly_value))

        overall_values = self._overall_norm(overall_values)

        return overall_values, new_poly_values, new_ray_values


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

        self._start_ray_size = 32
        self._ray_item_initial = nn.Linear(constants.n_ray_item_values,
                                           self._start_ray_size,
                                           bias=False)
        self._ray_item_initial_for_ray = nn.Linear(constants.n_ray_item_values,
                                                   self._start_ray_size,
                                                   bias=False)

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
        self._end_ray_size = self._start_ray_size

        overall_sizes = interpolate_sizes(self._initial_overall_size,
                                          self._end_fused_size,
                                          self._n_fused_blocks)
        poly_sizes = interpolate_sizes(self._start_poly_size,
                                       self._end_poly_size,
                                       self._n_fused_blocks)
        ray_sizes = interpolate_sizes(self._start_ray_size, self._end_ray_size,
                                      self._n_fused_blocks)
        self._fused_blocks = nn.ModuleList([
            FusedBlock(inp_overall, out_overall, inp_poly, out_poly, inp_ray,
                       out_ray)
            for ((inp_overall, out_overall), (inp_poly, out_poly),
                 (inp_ray,
                  out_ray)) in zip(overall_sizes, poly_sizes, ray_sizes)
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

        ray_values = []
        for ray_input in inputs.ray_inputs:
            print(ray_input.values.size())
            values = self._ray_item_initial(ray_input.values)
            print(values.size())
            print(ray_input.is_ray.size())
            values[ray_input.is_ray] = self._ray_item_initial_for_ray(
                ray_input.values)[ray_input.is_ray]
            ray_values.append((ray_input, self._activation(values)))

        overall_values = self._activation(
            self._overall_initial(
                torch.cat((inputs.overall_scene_features,
                           tri_values.view(tri_values.size(0), -1)),
                          dim=-1)))

        for block in self._initial_overall_blocks:
            overall_values = block(overall_values)

        for block in self._fused_blocks:
            (overall_values, poly_values,
             ray_values) = block(overall_values, poly_values, ray_values)

        # done with poly and ray
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
