import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, restore=False):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x, restore=restore)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx, restore=restore)
            return x, logpx

    def inverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
            return y, logpy


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None):
        return self.flow.inverse(x, logpx)

    def inverse(self, y, logpy=None):
        return self.flow.forward(y, logpy)


class SequentialFlowToy(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList, input_dim):
        super(SequentialFlowToy, self).__init__()
        self.chain = nn.ModuleList(layersList)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, logpx=None, restore=False):
        y = self.fc(x)

        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x, restore=restore)
            return x, y
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx, restore=restore)
            return (x, logpx), y

    def inverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
            return y, logpy
