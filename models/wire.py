import numpy as np
import torch
from torch import nn

from models.inr import INR


class ComplexGaborLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega_0: Frequency of Gabor sinusoid term
        sigma_0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=10.0,
        sigma_0=40.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma_0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class Wire(INR):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        non_linearity = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))

        first_layer = non_linearity(
            in_features=in_features,
            out_features=hidden_features,
            is_first=True,
            omega_0=first_omega_0,
            sigma_0=scale,
        )
        final_layer = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)

        super().__init__(
            non_linearity=non_linearity,
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            pos_encode=pos_encode,
            sidelength=sidelength,
            fn_samples=fn_samples,
            use_nyquist=use_nyquist,
            first_layer=first_layer,
            final_layer=final_layer,
        )

    def forward(self, coords):
        output = super().forward(coords)
        return output.real
