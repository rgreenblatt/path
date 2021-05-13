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

        self._norm = nn.LayerNorm(output_size)

    def forward(self, x):
        padding = torch.zeros(*x.size()[:-1],
                              self._pad_size,
                              device=x.device,
                              dtype=x.dtype)
        padded_input = torch.cat((x, padding), -1)
        # TODO: take a look at this!!!
        x = self._activation(self._contract(self._activation(self._expand(x))))
        return self._norm(padded_input + x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self._activation = nn.CELU()
        self._input_size = 24
        self._start_size = 128
        self._end_size = 512

        self._input_expand = nn.Linear(self._input_size, self._start_size)
        self._final = nn.Linear(self._end_size, self._end_size)

        self._n_blocks = 6

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
                                               self._end_size)
        self._coords_to_addr = nn.Linear(self._coord_block_size,
                                         self._end_size)

        self._output_block_size = 64
        self._output_size = 3

        self._output = nn.Linear(self._end_size, self._output_block_size)
        self._output_block1 = DenseBlock(self._output_block_size,
                                         self._output_block_size)
        self._output_block2 = DenseBlock(self._output_block_size,
                                         self._output_block_size)
        self._project_output = nn.Linear(self._output_block_size,
                                         self._output_size)
        self._output_activation = nn.CELU(alpha=0.05)

    def forward(self, scenes, coords):
        x = self._activation(self._input_expand(scenes))
        for block in self._scene_blocks:
            x = block(x)
        x = self._final(x)

        y = self._coords_block(self._activation(self._coords_expand(coords)))
        multiplier = torch.sigmoid(self._coords_to_multiplier(y))

        out = self._activation(
            self._output(
                torch.unsqueeze(x, 1) * multiplier +
                self._activation(self._coords_to_addr(y))))
        out = self._output_block1(out)
        out = self._output_block2(out)
        return self._output_activation(self._project_output(out))
