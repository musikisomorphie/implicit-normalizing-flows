import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['ActNorm1d', 'ActNorm2d']


class ActNormNd(nn.Module):

    def __init__(self, num_features, zero_pad=0, eps=1e-12):
        super(ActNormNd, self).__init__()
        assert zero_pad >= 0
        self.num_features = num_features - zero_pad
        self.zero_pad = zero_pad
        self.eps = eps

        self.weight = Parameter(torch.Tensor(self.num_features))
        self.bias = Parameter(torch.Tensor(self.num_features))
        self.register_buffer('initialized', torch.tensor(0))
        # print(self.weight.shape, self.bias.shape)

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None, restore=None):
        if self.zero_pad:
            x_pad, x = x[:, :self.zero_pad], x[:, self.zero_pad:]

        c = x.size(1)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(
                    batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)

        y = (x + bias.to(x)) * torch.exp(weight.to(x))
        if self.zero_pad:
            y = torch.cat((x_pad, y), dim=1)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        if self.zero_pad:
            y_pad, y = y[:, :self.zero_pad], y[:, self.zero_pad:]

        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if self.zero_pad:
            x = torch.cat((y_pad, x), dim=1)

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x[:, self.zero_pad:])

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.shape).contiguous().view(x.shape[0], -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
