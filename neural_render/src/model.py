import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self._activation = nn.CELU()
        self._hidden_size_multiplier = 4
        self._hidden_size = output_size * self._hidden_size_multiplier
        self._pad_size = output_size - input_size
        assert self._pad_size >= 0

        self._expand = nn.Linear(input_size, self._hidden_size)
        self._contract = nn.Linear(self._hidden_size, output_size)

        self._mul_divider = 16
        self._mul_size = output_size // self._mul_divider
        self._output_mul_size = self._mul_size * self._mul_divider
        self._contract_for_mul = nn.Linear(output_size, self._mul_size)

        self._norm = nn.LayerNorm(output_size)

    def forward(self, x):
        padding = torch.zeros(*x.size()[:-1],
                              self._pad_size,
                              device=x.device,
                              dtype=x.dtype)
        padded_input = torch.cat((x, padding), -1)
        x = self._activation(self._contract(self._activation(self._expand(x))))

        x_for_mul = x[..., :self._output_mul_size]
        x_not_mul = x[..., self._output_mul_size:]

        sub_shape = x_for_mul.size()[:-1]
        multiplier = torch.tanh(self._contract_for_mul(x)).unsqueeze(-1)
        x_mul = multiplier * x_for_mul.view(*sub_shape, self._mul_size, -1)

        x = torch.cat((x_mul.view(*sub_shape, -1), x_not_mul), -1)

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

        self._scene_blocks = nn.ModuleList()

        per_block_size = (self._end_size - self._start_size) / self._n_blocks
        last_size = self._start_size
        for i in range(self._n_blocks):
            new_block_size = round(per_block_size * (i + 1) + self._start_size)
            self._scene_blocks.append(DenseBlock(last_size, new_block_size))
            last_size = new_block_size
        assert last_size == self._end_size

        self._coord_size = 2
        self._coord_block_size = 32
        self._coords_expand = nn.Linear(self._coord_size,
                                        self._coord_block_size)
        self._coords_block = DenseBlock(self._coord_block_size,
                                        self._coord_block_size)
        self._coords_to_multiplier = nn.Linear(self._coord_block_size,
                                               self._multiplier_size)

        self._output_size = 3

        self._output = nn.Linear(self._multiplier_size, self._output_size)

        negative_allowance = 0.002
        self._output_activation = nn.CELU(alpha=negative_allowance)

    def forward(self, scenes, coords):
        x = self._activation(self._input_expand(scenes))
        for block in self._scene_blocks:
            x = block(x)
        x = self._activation(self._final(x))

        y = self._coords_block(self._activation(self._coords_expand(coords)))
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))

        return self._output_activation(
            self._output(torch.unsqueeze(x, 1) * multiplier))
