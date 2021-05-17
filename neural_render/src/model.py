import torch
from torch import nn
import torch.nn.functional as F


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
        self._start_size = 384
        self._end_size = 384
        self._multiplier_size = 384

        self._input_expand = nn.Linear(self._input_size, self._start_size)
        self._final = nn.Linear(self._end_size, self._multiplier_size)

        self._n_blocks = 12

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

        self._coords_to_addr = nn.Linear(self._coord_block_size,
                                         self._output_block_size)

        self._to_output_block = nn.Linear(self._multiplier_size,
                                          self._output_block_size)
        self._output_block = ResBlock(self._output_block_size,
                                      self._output_block_size)

        self._output_size = 3

        self._output = nn.Linear(self._output_block_size, self._output_size)

    def forward(self, scenes, coords):
        x = self._activation(self._input_expand(scenes))
        for block in self._scene_blocks:
            x = block(x)
        x = self._activation(self._final(x))

        y = self._coords_block(self._activation(self._coords_expand(coords)))
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))
        out = self._to_output_block(torch.unsqueeze(x, 1) * multiplier)

        return torch.relu(self._output(self._output_block(out)))


if __name__ == "__main__":
    assert len(list(interpolate_sizes(128, 507, 8))) == 8
    assert len(list(interpolate_sizes(507, 507, 8))) == 8
    for inp_s, out_s in interpolate_sizes(128, 507, 8):
        print("inp:", inp_s, "out:", out_s)
