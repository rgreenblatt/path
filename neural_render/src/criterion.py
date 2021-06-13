from torch import nn


class PerceptualLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.reduction = reduction
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, inp, target):
        vals = (self.l1(inp, target) / (target + 1.0))
        if self.reduction == 'mean':
            return vals.mean()
        elif self.reduction == 'sum':
            return vals.sum()
        elif self.reduction == 'none':
            return vals
        else:
            raise RuntimeError('invalid reduction!')
