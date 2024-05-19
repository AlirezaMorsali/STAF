import matplotlib.pyplot as plt
import numpy as np
import torch

from modules.models import INR


# Define the functions for STAF, SIREN, and WIRE
def paracnet(x, model_path):
    model = torch.load(model_path)
    results = []
    for i in range(len(model.net) - 1):
        y = (
            model.net[i]
            .apply_activation(torch.tensor(x).unsqueeze(0).to("cuda"))
            .squeeze()
        )
        results.append(y.detach().cpu().numpy())
    return results


def siren(x, omega=30):
    return np.sin(omega * x)


def wire(x, omega=10, sigma=3):
    return np.exp(1j * omega * x - np.abs(sigma * x) ** 2)


# Create the x values
x = np.linspace(-1, 1, 400)

# Generate the ParAcNet layers
paracnet_layers = paracnet(x, "data/staf.pth")

# Create the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot ParAcNet layers
for i, layer in enumerate(paracnet_layers):
    axs[0].plot(x, layer, label=f"Layer {i}")
axs[0].set_title(r"$\bf{STAF}$ $\sum_{i=0}^N c_i \sin(\omega_i x + \phi_i)$")
axs[0].set_xlabel("x")
axs[0].set_ylabel(r"$\sigma(x)$")
axs[0].legend()
axs[0].grid(True)  # Add grid


# Plot SIREN
axs[1].plot(x, siren(x))
axs[1].set_title(r"SIREN $\sin(\omega x)$")
axs[1].set_xlabel("x")
axs[1].set_ylabel(r"$\sigma(x)$")
axs[1].grid(True)  # Add grid

# Plot WIRE
axs[2].plot(x, wire(x).real)
axs[2].set_title(r"WIRE $e^{j\omega x}e^{-|\sigma x|^2}$")
axs[2].set_xlabel("x")
axs[2].set_ylabel(r"Re$\{\sigma(x)\}$")
axs[2].grid(True)  # Add grid

# Adjust layout
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig("data/staf_comparison_graphs.pdf")
plt.savefig("data/staf_comparison_graphs.png")

# Show the plot (optional)
plt.show()
