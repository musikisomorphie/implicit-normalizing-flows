import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(2. * math.pi * x) / math.pi * 0.5


class Identity(nn.Module):

    def forward(self, x):
        return x


class Zero(nn.Module):

    def forward(self, x):
        return torch.zeros_like(x).to(x)


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


class MaxMin(nn.Module):

    def forward(self, x):
        b, d = x.shape
        max_vals = torch.max(x.view(b, d // 2, 2), 2)[0]
        min_vals = torch.min(x.view(b, d // 2, 2), 2)[0]
        return torch.cat([max_vals, min_vals], 1)


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1).to(x) * (x - 2 / 3) + (x <= -1).to(x) * (x + 2 / 3) + ((x > -1) * (x < 1)).to(x) * x**3 / 3


class SwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta):
        beta_sigm = torch.sigmoid(beta * x)
        output = x * beta_sigm
        ctx.save_for_backward(x, output, beta)
        return output / 1.1

    @staticmethod
    def backward(ctx, grad_output):
        x, output, beta = ctx.saved_tensors
        beta_sigm = output / x
        grad_x = grad_output * \
            (beta * output + beta_sigm * (1 - beta * output))
        grad_beta = torch.sum(
            grad_output * (x * output - output * output)).expand_as(beta)
        return grad_x / 1.1, grad_beta / 1.1


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta.to(x)))).div_(1.1)


class PadZero(nn.Module):
    def __init__(self, left_pad=0, right_pad=0):
        super(PadZero, self).__init__()
        assert left_pad >= 0 and right_pad >= 0
        assert isinstance(left_pad, int) and isinstance(right_pad, int)
        self.left_pad = left_pad
        self.right_pad = right_pad

    def forward(self, x):
        pad_shape = list(x.shape)
        if self.left_pad:
            pad_shape[1] = self.left_pad
            x_left = torch.zeros(pad_shape).to(x)
            x = torch.cat((x_left, x), dim=1)

        if self.right_pad:
            pad_shape[1] = self.right_pad
            x_right = torch.zeros(pad_shape).to(x)
            x = torch.cat((x, x_right), dim=1)
            
        # print('pad_shape', pad_shape)
        return x

if __name__ == '__main__':

    m = Swish()
    xx = torch.linspace(-5, 5, 1000).requires_grad_(True)
    yy = m(xx)
    dd, dbeta = torch.autograd.grad(yy.sum() * 2, [xx, m.beta])

    import matplotlib.pyplot as plt

    plt.plot(xx.detach().numpy(), yy.detach().numpy(), label='Func')
    plt.plot(xx.detach().numpy(), dd.detach().numpy(), label='Deriv')
    plt.plot(xx.detach().numpy(), torch.max(dd.detach().abs() - 1,
                                            torch.zeros_like(dd)).numpy(), label='|Deriv| > 1')
    plt.legend()
    plt.tight_layout()
    plt.show()
