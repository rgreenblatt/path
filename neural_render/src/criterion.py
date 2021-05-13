from torch import nn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, inp, target):
        return (self.l1(inp, target) / (target + 1.0)).mean()
