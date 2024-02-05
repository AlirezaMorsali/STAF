import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import models, utils

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
# plt.gray()


import os
import sys
from tqdm import tqdm
import importlib
import time
from datetime import datetime

import numpy as np
from scipy import io

import matplotlib.pyplot as plt

# plt.gray()

import cv2
from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim

from modules import models
from modules import utils
import wandb


parser = argparse.ArgumentParser(description="Image reconstruction parameters")
parser.add_argument(
    "-i",
    "--input_image",
    type=str,
    help="Path to input image",
    default="./data/cameraman.tif",
)
parser.add_argument(
    "-n",
    "--nonlinearity",
    choices=["wire", "siren", "mfn", "relu", "posenc", "gauss"],
    type=str,
    help="Name of nonlinearity",
    # default="siren",
    # default="wire",
    default="parac",
)
parser.add_argument(
    "-s",
    "--resize",
    type=int,
    default=None,
    help="If not None resize to the provided size",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=250,
    help="Number of epochs",
)

parser.add_argument(
    "-dr",
    "--dryrun",
    action="store_true",
    default=False,
    help="Number of epochs",
)


args = parser.parse_args()

if not args.dryrun:
    now = datetime.now()
    formatted_date_time_modified = now.strftime("%d_%M")
    run_name = f"activation_{args.nonlinearity}_{now.strftime('%d_%M')}"
    run = wandb.init(entity="pracnet", project="parametric", name=run_name)

image_path = args.input_image
nonlin = args.nonlinearity
# "posenc"  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'

niters = args.epochs  # Number of SGD iterations
learning_rate = 1e-3  # Learning rate.

# WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
# MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

tau = 3e7  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
noise_snr = 2  # Readout noise (dB)

# Gabor filter constants.
# We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
omega0 = 30.0  # Frequency of sinusoid
sigma0 = 4.0  # Sigma of Gaussian

# Network parameters
hidden_layers = 2  # Number of hidden layers in the MLP
hidden_features = 256  # Number of hidden units per layer
maxpoints = 256 * 256  # Batch size

# Read image and scale. A scale of 0.5 for parrot image ensures that it
# fits in a 12GB GPU
input_image = plt.imread(image_path)
if args.resize:
    input_image = cv2.resize(input_image, (args.resize, args.resize))
im = utils.normalize(input_image.astype(np.float32), True)
# im = cv2.resize(im, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)

imsh = im.shape
(
    H,
    W,
) = (
    imsh[0],
    imsh[1],
)
if len(imsh) == 2:
    imdim = 1
    im = im[:, :, np.newaxis]
else:
    imdim = 3

# Create a noisy image
# im_noisy = utils.measure(im, noise_snr, tau)
im_noisy = im

if nonlin == "posenc":
    nonlin = "relu"
    posencode = True

    if tau < 100:
        sidelength = int(max(H, W) / 3)
    else:
        sidelength = int(max(H, W))

else:
    posencode = False
    sidelength = H

model = models.get_INR(
    nonlin=nonlin,
    in_features=2,
    out_features=imdim,
    hidden_features=hidden_features,
    hidden_layers=hidden_layers,
    first_omega_0=omega0,
    hidden_omega_0=omega0,
    scale=sigma0,
    pos_encode=posencode,
    sidelength=sidelength,
)


# Send model to Device
model.to(device)

x = torch.linspace(-1, 1, 1000)
for i, net in enumerate(model.net):
    if isinstance(net, model.nonlin):
        y = net.apply_activation(x)
        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log(
            {
                f"initial_activation_layer_{i}": wandb.plot.line(
                    table, "x", "y", title=f"Initial Parametric Activation Layer {i}"
                )
            }
        )


print("Number of parameters: ", utils.count_parameters(model))
print("Input PSNR: %.2f dB" % utils.psnr(im, im_noisy))

# Create an optimizer
# optim = torch.optim.Adam(
#     lr=learning_rate * min(1, maxpoints / (H * W)), params=model.parameters()
# )

optim = torch.optim.Adam(
    lr=1e-4, betas=(0.9, 0.999), eps=1e-08, params=model.parameters()
)

# Schedule to reduce lr to 0.1 times the initial rate in final epoch
scheduler = LambdaLR(optim, lambda x: 0.1 ** min(x / niters, 1))

x = torch.linspace(-1, 1, W)
y = torch.linspace(-1, 1, H)

X, Y = torch.meshgrid(x, y, indexing="xy")
coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

image_tensor = torch.tensor(im)
image_tensor.to(device)
gt = image_tensor.reshape(H * W, imdim)[None, ...]
gt_tensor = torch.tensor(im_noisy)
gt_tensor.to(device)
gt_noisy = gt_tensor.reshape(H * W, imdim)[None, ...]

mse_array = torch.zeros(niters, device=device)
mse_loss_array = torch.zeros(niters, device=device)
time_array = torch.zeros_like(mse_array)

best_mse = torch.tensor(float("inf"))
best_img = None

rec = torch.zeros_like(gt)

tbar = tqdm(range(niters))
init_time = time.time()

b_coords = coords.to(device)

for epoch in tbar:
    # indices = torch.randperm(H * W)
    # for b_idx in range(0, H * W, maxpoints):
    # b_indices = indices[b_idx : min(H * W, b_idx + maxpoints)]
    # b_coords = coords[:, b_indices, ...].to(device)
    # b_indices = b_indices.to(device)
    pixelvalues = model(b_coords)

    with torch.no_grad():
        rec = pixelvalues

    loss = ((pixelvalues - gt_noisy) ** 2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    time_array[epoch] = time.time() - init_time

    with torch.no_grad():
        mse_loss_array[epoch] = ((gt_noisy - rec) ** 2).mean().item()
        mse_array[epoch] = ((gt - rec) ** 2).mean().item()
        im_gt = gt.reshape(H, W, imdim).permute(2, 0, 1)[None, ...]
        im_rec = rec.reshape(H, W, imdim).permute(2, 0, 1)[None, ...]

        psnrval = -10 * torch.log10(mse_array[epoch])
        tbar.set_description("%.1f" % psnrval)
        tbar.refresh()

    scheduler.step()

    imrec = rec[0, ...].reshape(H, W, imdim).detach().cpu().numpy()

    cv2.imshow("Reconstruction", imrec[..., ::-1])
    cv2.waitKey(1)

    if (mse_array[epoch] < best_mse) or (epoch == 0):
        best_psnr = psnrval
        best_mse = mse_array[epoch]
        best_img = imrec

    if not args.dryrun:
        run.log({"loss": loss, "psnr": psnrval})

if posencode:
    nonlin = "posenc"

mdict = {
    "rec": best_img,
    "gt": im,
    "im_noisy": im_noisy,
    "mse_noisy_array": mse_loss_array.detach().cpu().numpy(),
    "mse_array": mse_array.detach().cpu().numpy(),
    "time_array": time_array.detach().cpu().numpy(),
}

os.makedirs("results/denoising", exist_ok=True)
io.savemat("results/denoising/%s.mat" % nonlin, mdict)
cv2.imwrite(f"results/denoising/{nonlin}.jpg", best_img[..., ::-1])

print("Best PSNR: %.2f dB" % utils.psnr(im, best_img))  # %%

if not args.dryrun:
    img_estim_cpu = best_img[..., ::-1]
    wandb.log(
        {f"{nonlin}_ct": [wandb.Image(img_estim_cpu, caption=f"rec_image {nonlin}")]}
    )

    x = torch.linspace(-1, 1, 1000)
    for i, net in enumerate(model.net):
        if isinstance(net, model.nonlin):
            y = net.apply_activation(x)
            data = [[x, y] for (x, y) in zip(x, y)]
            table = wandb.Table(data=data, columns=["x", "y"])
            wandb.log(
                {
                    f"trained_activation_layer_{i}": wandb.plot.line(
                        table,
                        "x",
                        "y",
                        title=f"Trained Parametric Activation Layer {i}",
                    )
                }
            )


fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].imshow(input_image)
axes[1].imshow(best_img)
axes[0].title.set_text("Original")
axes[1].title.set_text(f"{nonlin}")
plt.savefig(f"results/Original-vs-{nonlin}.png")  # Save as PNG image

plt.gray()
plt.show()
