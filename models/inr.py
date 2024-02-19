import torch
from torch import nn


class INR(nn.Module):
    def __init__(
        self,
        non_linearity,
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
        first_layer=None,
        final_layer=None,
    ):
        super().__init__()

        self.non_linearity = non_linearity
        self.pos_encode = pos_encode
        self.net = []

        # first layer
        if first_layer:
            self.net.append(first_layer)
        else:
            first_layer = self.non_linearity(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )

            self.net.append(first_layer)

        # hidden layers
        self.net.extend(
            [
                self.non_linearity(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    sigma_0=scale,
                )
                for _ in range(hidden_layers)
            ]
        )

        # final layer
        if final_layer:
            # if the model does have some sort of special final layer
            self.net.append(final_layer)
        else:
            # otherwise, the final layer will be a simple Linear layer
            final_linear = nn.Linear(hidden_features, out_features, dtype=torch.float)
            self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # if self.pos_encode:
        #     coords = self.positional_encoding(coords)

        output = self.net(coords)
        return output
