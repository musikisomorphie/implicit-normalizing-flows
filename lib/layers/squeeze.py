import torch
import torch.nn as nn

__all__ = ['SqueezeLayer']


class SqueezeLayer(nn.Module):

    def __init__(self, shuffle_factor):
        super(SqueezeLayer, self).__init__()
        self.shuffle_factor = shuffle_factor

    def forward(self, x, logpx=None, restore=False):
        squeeze_x = torch.pixel_unshuffle(x, self.shuffle_factor)
        if logpx is None:
            return squeeze_x
        else:
            logpx[1] = torch.pixel_unshuffle(logpx[1], self.shuffle_factor)
            return squeeze_x, logpx

    def inverse(self, y, logpy=None):
        unsqueeze_y = torch.pixel_shuffle(y, self.shuffle_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            logpy[1] = torch.pixel_shuffle(logpy[1], self.shuffle_factor)
            return unsqueeze_y, logpy
