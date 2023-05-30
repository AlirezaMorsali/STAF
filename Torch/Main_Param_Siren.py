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
from lib import torchnets, torchutils




"""Let's instantiate the dataset and our Siren. As pixel coordinates are 2D, the siren has 2 input features, and since the image is grayscale, it has one output channel."""
imsize = 256
nf = 3
classic = 1
cameraman = torchutils.ImageFitting(imsize)
dataloader = DataLoader(cameraman,
                        # batch_size=256*256,
                        batch_size=1,
                        pin_memory=False,
                        num_workers=0,
                        shuffle=False)

img_para = torchnets.Parasin(in_features=2,
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        nf=nf,
        outermost_linear=True)

img_siren = torchnets.Siren(in_features=2,
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
img_para.cpu()
"""We now fit Siren in a simple training loop. Within only hundreds of iterations, the image and its gradients are approximated well."""

total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 1

# optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

optim = torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=img_siren.parameters())
optim_para = torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=img_para.parameters())

# optim = torch.optim.SGD(lr=0.001, momentum=0, params=img_siren.parameters())
# optim_para = torch.optim.SGD(lr=0.001, momentum=0, params=img_para.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cpu(), ground_truth.cpu()
psnr = []
psnr_para = []

for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    model_output_para, coords_para = img_para(model_input)
    # model_output_para, coords_para = img_siren(model_input)
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
axes[2].imshow(outim_para.view(imsize, imsize).detach().numpy(), cmap='gray')
axes[3].plot(psnr)
axes[3].plot(psnr_para)
axes[3].legend(["Siren", "Param"])
axes[3].grid()
axes[0].title.set_text('Original')
axes[1].title.set_text('Siren')
axes[2].title.set_text('Param')
axes[3].title.set_text('PSNR Torch')
fig.suptitle('Torch', fontsize=16)

plt.show()

# recim = outim.view(imsize, imsize).detach().numpy()
# with open(f"data_Siren_torch.data", "wb") as f:
#     pickle.dump((psnr, recim), f)
