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
from Torch import nrnets

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)

    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=False,
                 first_omega_0=30.,
                 hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(in_features,
                      hidden_features,
                      is_first=True,
                      omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features,
                          hidden_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features,
                          out_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords


# def get_cameraman_tensor(sidelength):
#     Image1 = skimage.data.camera().astype(np.float32)
#     Image1 = Image1.reshape(1, Image1.shape[0], Image1.shape[1]) / 255
#     img = tensor(Image1)
#     return img


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        # ToTensor()
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)

    return img


"""<a id='section_1'></a>
## Fitting an image

First, let's simply fit that image!

We seek to parameterize a greyscale image $f(x)$ with pixel coordinates $x$ with a SIREN $\Phi(x)$.

That is we seek the function $\Phi$ such that:
$\mathcal{L}=\int_{\Omega} \lVert \Phi(\mathbf{x}) - f(\mathbf{x}) \rVert\mathrm{d}\mathbf{x}$
 is minimized, in which $\Omega$ is the domain of the image.

We write a little datast that does nothing except calculating per-pixel coordinates:
"""


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


"""Let's instantiate the dataset and our Siren. As pixel coordinates are 2D, the siren has 2 input features, and since the image is grayscale, it has one output channel."""
imsize = 512
classic = 1
cameraman = ImageFitting(imsize)
dataloader = DataLoader(cameraman,
                        # batch_size=256*256,
                        batch_size=1,
                        pin_memory=False,
                        num_workers=0,
                        shuffle=False)
if classic:
    img_siren = nrnets.Parasin(in_features=2,
                out_features=1,
                hidden_features=256,
                hidden_layers=3,
                outermost_linear=True)
else:
    img_siren = Siren(in_features=2,
                out_features=1,
                hidden_features=256,
                hidden_layers=3,
                outermost_linear=True)

img_siren.cpu()
"""We now fit Siren in a simple training loop. Within only hundreds of iterations, the image and its gradients are approximated well."""

total_steps = 10  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 1

# optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
optim = torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=img_siren.parameters())
# optim = torch.optim.SGD(lr=0.01, momentum=0, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cpu(), ground_truth.cpu()
psnr = []

for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    loss = ((model_output - ground_truth)**2).mean()
    outim = model_output.cpu().detach().view(-1, 1)
    psnr.append(10*torch.log10(1 / torch.mean((outim - cameraman.pixels)**2)))
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f, psnr %0.6f" % (step, loss, psnr[-1]))
    optim.zero_grad()
    loss.backward()
    optim.step()



fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cameraman.pixels.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[1].imshow(outim.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[2].plot(psnr)
axes[2].grid()
axes[0].title.set_text('Original')
axes[1].title.set_text('Reconstructed Torch')
axes[2].title.set_text('PSNR Torch')
fig.suptitle('Torch', fontsize=16)
#
plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(cameraman.pixels.view(imsize, imsize).detach().numpy(), cmap='gray')
# axes[1].imshow(outim.view(imsize, imsize).detach().numpy(), cmap='gray')
# axes[2].plot(psnr)
# axes[2].grid()
recim = outim.view(imsize, imsize).detach().numpy()
with open(f"data_Siren_torch.data", "wb") as f:
    pickle.dump((psnr, recim), f)
