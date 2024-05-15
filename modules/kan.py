import torch
from torch import nn

from modules.encoding import Encoding

from .ChebyKANLayer import ChebyKANLayer


class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(1, 8, 8)
        self.chebykan2 = ChebyKANLayer(8, 1, 8)

    def forward(self, x):
        x = self.chebykan1(x)
        x = self.chebykan2(x)
        return x


class INR(nn.Module):
    def __init__(
        self, in_features, hidden_features, hidden_layers, out_features, degree=8
    ):
        super().__init__()

        self.net = []
        self.net.append(ChebyKANLayer(in_features, hidden_features, degree))
        for i in range(hidden_layers):
            self.net.append(ChebyKANLayer(hidden_features, hidden_features, degree))

        self.net.append(ChebyKANLayer(hidden_features, out_features, degree))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        output = self.net(coords)

        return output