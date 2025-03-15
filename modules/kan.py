import torch
from torch import nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = (
                2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
            )
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", cheby, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

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