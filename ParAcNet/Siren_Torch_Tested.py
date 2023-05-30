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
img_para = nrnets.Parasin(in_features=2,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True)
img_siren = Siren(in_features=2,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True)


# img_siren = Siren(in_features=2,
#                   out_features=1,
#                   hidden_features=256,
#                   hidden_layers=3,
#                   outermost_linear=True)
img_siren.cpu()
"""We now fit Siren in a simple training loop. Within only hundreds of iterations, the image and its gradients are approximated well."""

total_steps = 10  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 1

# optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
optim = torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=img_siren.parameters())
optim_para = torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=img_para.parameters())
# optim = torch.optim.SGD(lr=0.01, momentum=0, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cpu(), ground_truth.cpu()
psnr = []
psnr_para = []

for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    model_output_para, coords_para = img_para(model_input)
    loss = ((model_output - ground_truth)**2).mean()
    loss_para = ((model_output_para - ground_truth)**2).mean()
    outim = model_output.cpu().detach().view(-1, 1)
    outim_para = model_output_para.cpu().detach().view(-1, 1)
    psnr.append(10*torch.log10(1 / torch.mean((outim - cameraman.pixels)**2)))
    psnr_para.append(10*torch.log10(1 / torch.mean((outim_para - cameraman.pixels)**2)))
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f, psnr %0.6f" % (step, loss, psnr[-1]))
        print("Step %d, Total loss %0.6f, psnr %0.6f" % (step, loss_para, psnr_para[-1]))
    optim.zero_grad()
    optim_para.zero_grad()
    loss.backward()
    loss_para.backward()
    optim.step()
    optim_para.step()



fig, axes = plt.subplots(1, 4, figsize=(18, 6))
axes[0].imshow(cameraman.pixels.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[1].imshow(outim.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[2].imshow(outim.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[3].plot(psnr)
axes[3].plot(psnr_para)
axes.legend(["Siren", "Param"])
axes[2].grid()
axes[0].title.set_text('Original')
axes[1].title.set_text('Siren')
axes[2].title.set_text('Param')
axes[3].title.set_text('PSNR Torch')
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
