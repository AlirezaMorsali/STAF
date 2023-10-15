import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from lib import torchutils

class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

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
        temp = torch.sin(self.omega_0 * self.linear(input))
        # print(f'linear_size:{temp.shape}')
        return temp

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords


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


class ParaLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, nf, bias=True, is_first=False, omega_0=30
    ):
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

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate), intermediate


class Parasin(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        nf,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            ParaLayer(
                in_features, hidden_features, is_first=True, nf=nf, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                ParaLayer(
                    hidden_features,
                    hidden_features,
                    nf=nf,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                ParaLayer(
                    hidden_features,
                    out_features,
                    nf=nf,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        # print('here')
        output = self.net(coords)

        return output, coords


# a = torch.ones(1, 2, 2)

# model = Parasin(
#     in_features=2,
#     out_features=1,
#     hidden_features=256,
#     hidden_layers=3,
#     outermost_linear=True,
# )
# img_siren = Siren(in_features=2,
#         out_features=1,
#         hidden_features=256,
#         hidden_layers=3,
#         outermost_linear=True)

# img_siren.cpu()

# imsize = 512
# classic = 1
# cameraman = torchutils.ImageFitting(imsize)
# dataloader = DataLoader(
#     cameraman,
#     # batch_size=256*256,
#     batch_size=1,
#     pin_memory=False,
#     num_workers=0,
#     shuffle=False,
# )

# inp = next(iter(dataloader))
# inp = inp[0]

# img_siren(inp)
# model(inp)
# # model_output_para, coords_para = model(inp)
