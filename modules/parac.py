#!/usr/bin/env python3
#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize


def param_act(linout, ws, bs, phis):
    linoutx = linout.unsqueeze(-1).repeat_interleave(ws.shape[0], dim=3)
    wsx = ws.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
    bsx = bs.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
    phisx = phis.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
    temp = bsx * (torch.sin((wsx * linoutx) + phisx))
    temp2 = torch.sum(temp, 3)
    # print(f"Activation Call input size:{linout.shape}")
    # print(f"Activation Call output size:{temp2.shape}")
    return temp2


class Parasin(nn.Module):
    """
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    discussion of omega_0.

    If is_first=True, omega_0 is a frequency factor which simply multiplies
    the activations before the nonlinearity. Different signals may require
    different omega_0 in the first layer - this is a hyperparameter.

    If is_first=False, then the weights will be divided by omega_0 so as to
    keep the magnitude of activations constant, but boost gradients to the
    weight matrix (see supplement Sec. 1.5)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        scale=10.0,
        init_weights=True,
    ):
        # def __init__(
        # self, in_features, out_features, nf, bias=True, is_first=False, omega_0=30
        # ):
        nf = 3
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.ws = nn.Parameter(torch.ones(nf), requires_grad=True)
        self.bs = nn.Parameter(torch.ones(nf), requires_grad=True)
        self.phis = nn.Parameter(torch.zeros(nf), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        # return torch.sin(self.omega_0 * self.linear(input))
        temp = self.omega_0 * self.linear(input)
        # print(temp.shape)
        return param_act(temp, self.ws, self.bs, self.phis)


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = Parasin

        self.net = []
        self.net.append(
            self.nonlin(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                scale=scale,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

            with torch.no_grad():
                const = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)

            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)

        output = self.net(coords)

        return output
