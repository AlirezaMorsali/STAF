import math

import torch
from torch import nn

from models import INR


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
        sigma_0=10.0,
        init_weights=True,
    ):
        super().__init__()

        self.nf = 5
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights and parameters
        self.init_params()
                
    def init_params(self):
        """
        Initializes the parameters for the sinusoidal activations.
        """
        ws = self.omega_0 * torch.rand(self.nf)
        self.ws = nn.Parameter(ws, requires_grad=True)

        # Initialize phases uniformly in the range [-π, π]
        self.phis = nn.Parameter(
            -math.pi + 2 * math.pi * torch.rand(self.nf), requires_grad=True
        )

        # Initialize scale factors based on a Laplace distribution
        diversity_y = 1 / (2 * self.nf)
        laplace_samples = torch.distributions.Laplace(0, diversity_y).sample((self.nf,))
        self.bs = nn.Parameter(torch.sign(laplace_samples) * torch.sqrt(torch.abs(laplace_samples)), requires_grad=True)

    def forward(self, input):
        return self.param_act(self.linear(input))

    def param_act(self, linout):
        sinusoidal_modulation = self.bs * torch.sin(self.ws * linout.unsqueeze(-1) + self.phis)
        return sinusoidal_modulation.sum(dim=-1)

    def apply_activation(self, x):
        y = torch.zeros_like(x)
        for i in range(len(self.ws)):
            y += self.bs[i] * torch.sin((self.ws[i] * x) + self.phis[i])
        return y


class Parac(INR):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=30,
        hidden_omega_0=30,
        scale=10,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        non_linearity = Parasin

        # final layer of this model, will be a simple linear
        final_layer = None
        super().__init__(
            non_linearity,
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            first_omega_0,
            hidden_omega_0,
            scale,
            pos_encode,
            sidelength,
            fn_samples,
            use_nyquist,
            final_layer=final_layer,
        )
