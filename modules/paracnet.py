#!/usr/bin/env python

import numpy as np
import torch
from torch import nn


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
        super().__init__()

        self.nf = 5
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # self.ws = nn.Parameter(torch.ones(self.nf), requires_grad=True)

        ws = omega_0 * torch.rand(self.nf)
        # ws = omega_0 * torch.ones(self.nf)
        # ws = torch.arange(15, 15 + self.nf).float()
        self.ws = nn.Parameter(ws, requires_grad=True)
        self.phis = nn.Parameter(requires_grad=True)
        self.bs = nn.Parameter(requires_grad=True)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            uniform_samples = torch.rand(self.nf)

            # Scale and shift the samples to the range [-π, π]
            lower_bound = -torch.tensor([3.14159265358979323846])  # -π
            upper_bound = torch.tensor([3.14159265358979323846])  # π
            scaled_samples = lower_bound + (upper_bound - lower_bound) * uniform_samples

            self.phis = nn.Parameter(scaled_samples, requires_grad=True)

            # Mean and diversity for Laplace random variable Y
            mean_y = 0
            diversity_y = 2 / (4 * self.nf)
            # Generate Laplace random variable Y
            laplace_samples = torch.distributions.laplace.Laplace(
                mean_y, diversity_y
            ).sample((self.nf,))

            # Compute C from Y
            c_samples = torch.sign(laplace_samples) * torch.sqrt(
                torch.abs(laplace_samples)
            )
            self.bs = nn.Parameter(c_samples, requires_grad=True)

    def siren_init_weights(self):
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
        # temp = self.omega_0 * self.linear(input)
        temp = self.linear(input)
        # print(temp.shape)
        return self.param_act(temp)

    def param_act(self, linout):
        ws, bs, phis = (self.ws, self.bs, self.phis)
        linoutx = linout.unsqueeze(-1).repeat_interleave(ws.shape[0], dim=3)
        wsx = ws.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
        bsx = bs.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
        phisx = phis.expand(linout.shape[0], linout.shape[1], linout.shape[2], -1)
        temp = bsx * (torch.sin((wsx * linoutx) + phisx))
        temp2 = torch.sum(temp, 3)
        # print(f"Activation Call input size:{linout.shape}")
        # print(f"Activation Call output size:{temp2.shape}")
        return temp2

    def apply_activation(self, x):
        y = torch.zeros_like(x)
        for i in range(len(self.ws)):
            y += self.bs[i] * torch.sin((self.ws[i] * x) + self.phis[i])
        return y


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
        output = self.net(coords)

        return output
