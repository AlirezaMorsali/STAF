#!/usr/bin/env python

import time
import os
import sys
import tqdm
from scipy import io
from dotenv import load_dotenv

load_dotenv()

import wandb
import numpy as np

import cv2
import matplotlib.pyplot as plt

plt.gray()

from skimage.metrics import structural_similarity as ssim_func

import torch
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils
from modules import lin_inverse

import argparse


# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT image parameters")
    parser.add_argument(
        "-i",
        "--input_image",
        type=str,
        help="Path to input image",
        default="./data/chest.png",
    )
    parser.add_argument(
        "-n",
        "--nonlinearity",
        choices=["wire", "siren", "mfn", "relu", "posenc", "gauss"],
        type=str,
        help="Name of nonlinearity",
        default="parac",
    )
    args = parser.parse_args()

    input_image = args.input_image
    nonlin = args.nonlinearity

    niters = 5000  # Number of SGD iterations
    learning_rate = 5e-3  # Learning rate.

    nmeas = 100  # Number of CT measurement

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    # Noise is not used in this script, but you can do so by modifying line 82 below
    tau = 3e1  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    omega0 = 10.0  # Frequency of sinusoid
    sigma0 = 10.0  # Sigma of Gaussian

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = 300  # Number of hidden units per layer

    # Generate sampling angles
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).cuda()

    # Create phantom
    image_name_ext = "chest.png"
    image_name = image_name_ext.split(".")[0]
    image_path = os.path.join("data", image_name_ext)

    img = cv2.imread(image_path).astype(np.float32)[..., 1]
    img = utils.normalize(img, True)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].cuda()

    if os.getenv("WANDB_LOG") in ["true", "True", True]:
        run_name = f'{nonlin}_{image_name}_ct__{str(time.time()).replace(".", "_")}'
        xp = wandb.init(
            name=run_name, project="pracnet", resume="allow", anonymous="allow"
        )

    # Create model
    if nonlin == "posenc":
        nonlin = "relu"
        posencode = True
    else:
        posencode = False

    model = models.get_INR(
        nonlin=nonlin,
        in_features=2,
        out_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        first_omega_0=omega0,
        hidden_omega_0=omega0,
        scale=sigma0,
        pos_encode=posencode,
        sidelength=nmeas,
    )

    model = model.cuda()

    with torch.no_grad():
        sinogram = lin_inverse.radon(imten, thetas).detach().cpu()
        sinogram = sinogram.numpy()
        sinogram_noisy = utils.measure(sinogram, noise_snr, tau).astype(np.float32)
        # Set below to sinogram_noisy instead of sinogram to get noise in measurements
        sinogram_ten = torch.tensor(sinogram).cuda()

    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()

    X, Y = torch.meshgrid(x, y, indexing="xy")

    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...].to(device)
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optimizer, lambda x: 0.1 ** min(x / niters, 1))

    best_loss = float("inf")
    loss_array = np.zeros(niters)
    best_im = None

    tbar = tqdm.tqdm(range(niters))
    result = torch.zeros_like(imten).to(device)

    for idx in tbar:
        # Estimate image
        img_estim = model(coords).reshape(-1, H, W)[None, ...]

        # Compute sinogram
        sinogram_estim = lin_inverse.radon(img_estim, thetas)

        loss = ((sinogram_ten - sinogram_estim) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            img_estim_cpu = img_estim.detach().cpu().squeeze().numpy()
            if sys.platform == "win32":
                cv2.imshow("Image", img_estim_cpu)
                cv2.waitKey(1)

            loss_gt = ((img_estim - imten) ** 2).mean()
            loss_array[idx] = loss_gt.item()

            if loss_gt < best_loss:
                best_loss = loss_gt
                best_im = img_estim

            tbar.set_description("%.4f" % (-10 * np.log10(loss_array[idx])))
            tbar.refresh()

            psnr2 = utils.psnr(img, img_estim_cpu)
            ssim2 = ssim_func(img, img_estim_cpu)

            if os.getenv("WANDB_LOG") in ["true", "True", True]:
                xp.log({"loss": loss.item(), "psnr": psnr2, "ssim": ssim2})

    img_estim_cpu = best_im.detach().cpu().squeeze().numpy()

    mdict = {
        "result": img_estim_cpu,
        "loss_array": loss_array,
        "sinogram": sinogram,
        "gt": img,
    }

    RESULTS_SAVE_PATH_BASE = os.path.join(
        os.getenv("RESULTS_SAVE_PATH"), f"{nonlin}_ct"
    )
    os.makedirs(RESULTS_SAVE_PATH_BASE, exist_ok=True)
    io.savemat(
        os.path.join(
            RESULTS_SAVE_PATH_BASE,
            "%s_%d.mat" % (nonlin, nmeas),
        ),
        mdict,
    )

    psnr2 = utils.psnr(img, img_estim_cpu)
    ssim2 = ssim_func(img, img_estim_cpu)

    print("PSNR: %.1f dB | SSIM: %.2f" % (psnr2, ssim2))

    # saving the model
    MODEL_SAVE_PATH_BASE = os.path.join(os.getenv("RESULTS_SAVE_PATH"), f"{nonlin}_ct")
    os.makedirs(
        MODEL_SAVE_PATH_BASE,
        exist_ok=True,
    )
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_SAVE_PATH_BASE, f"{nonlin}_{nmeas}.pth"),
    )

    print("saving the CT image on WANDB")
    wandb.log(
        {f"{nonlin}_ct": [wandb.Image(img_estim_cpu, caption=f"Ct image {nonlin}")]}
    )
