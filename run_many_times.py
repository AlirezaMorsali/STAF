N_TIMES = 10

import argparse
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import models, utils
from image_reconst import load_image_data, get_model

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


import os
from tqdm import tqdm
import time

import numpy as np
from scipy import io

import matplotlib.pyplot as plt

# plt.gray()

import cv2

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils


parser = argparse.ArgumentParser(description="Image reconstruction parameters")
parser.add_argument(
    "-i",
    "--input_image",
    type=str,
    help="Path to input image",
    default="./data/cameraman.jpg",
)
parser.add_argument(
    "-n",
    "--nonlinearity",
    choices=["wire", "siren", "mfn", "relu", "posenc", "gauss", 'parac'],
    type=str,
    help="Name of nonlinearity",
    # default="siren",
    # default="wire",
    default="parac",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate. Parac works best at 1e-4, Wire at 5e-3 to 2e-2, SIREN at 1e-3 to 2e-3.",
)
parser.add_argument(
    "-s",
    "--resize",
    type=int,
    default=None,
    help="If not None resize to the provided size",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Epcohs of maining", default=251
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=128 * 128,
    help="Batch size.",
)
parser.add_argument(
    "--live",
    type=bool,
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to show the reconstructed image live during the training or not.",
)

config_str = f"-n parac -b {64*64}"
args = parser.parse_args(config_str.split())
print(args)

def train_cvpr(image, args):
    nonlin = args.nonlinearity
    # "posenc"  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'

    niters = 251  # Number of SGD iterations
    learning_rate = 1e-4  # Learning rate.

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    tau = 3e7  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 30.0  # Frequency of sinusoid
    sigma0 = 4.0  # Sigma of Gaussian

    # Network parameters
    hidden_layers = 3  # Number of hidden layers in the MLP
    hidden_features = 256  # Number of hidden units per layer
    maxpoints = 256 * 256  # Batch size

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

    image_tensor = torch.tensor(im).to(device)
    gt = image_tensor.reshape(H * W, imdim)[None, ...].to(device)
    gt_tensor = torch.tensor(im_noisy).to(device)
    gt_tensor.to(device)
    gt_noisy = gt_tensor.reshape(H * W, imdim)[None, ...].to(device)

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

        # cv2.imshow("Reconstruction", imrec[..., ::-1])
        # cv2.waitKey(1)

        if (mse_array[epoch] < best_mse) or (epoch == 0):
            best_psnr = psnrval
            best_mse = mse_array[epoch]
            best_img = imrec

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

    best_psnr = utils.psnr(im, best_img)
    print("Best PSNR: %.2f dB" % best_psnr)  # %%

    # fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # axes[0].imshow(input_image)
    # axes[1].imshow(best_img)
    # axes[0].title.set_text("Original")
    # axes[1].title.set_text(f"{nonlin}")
    # plt.savefig(f"results/Original-vs-{nonlin}.png")  # Save as PNG image

    # plt.gray()
    # plt.show()

    return best_psnr

def train_main(args, img, wandb_xp=None):
    if args.nonlinearity == "wire":
        # Gabor filter constants.
        # We suggest omega0 = 4 and sigma0 = 4 for reconst, and omega0=20, sigma0=30 for image representation
        omega0 = 20.0  # Frequency of sinusoid
        sigma0 = 30.0  # Sigma of Gaussian

    else:
        omega0 = 30.0  # Frequency of sinusoid
        sigma0 = 4.0  # Sigma of Gaussian

    img = utils.normalize(img.astype(np.float32), fullnormalize=True)

    H, W = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # grayscale image
        img_dim = 1
        img = img[:, :, np.newaxis]
    else:
        # rgb image
        img_dim = 3

    model = get_model(
        non_linearity=args.nonlinearity,
        out_features=img_dim,
        hidden_features=256,
        hidden_layers=3,
        first_omega_0=omega0,
        hidden_omega_0=omega0,
        scale=sigma0,
    ).to(device)

    print("Number of parameters: ", utils.count_parameters(model))

    optim = torch.optim.Adam(
        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, params=model.parameters()
    )

    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    lr_sched = LambdaLR(optim, lambda x: 0.1 ** min(x / args.epochs, 1))

    X, Y = torch.meshgrid(
        torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy"
    )
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1))).to(device)

    gt = (
        torch.tensor(img).reshape(H * W, img_dim).to(device)
    )  # the model output will be of the shape 3 -> (x, y, z) - so ground truth also have to have shape of 3

    prog_bar = tqdm(range(args.epochs))
    psnr_vals = []

    for epoch in prog_bar:
        indices = torch.randperm(H * W).to(device)
        reconst_arr = torch.zeros_like(gt).to(device)

        # batch training
        train_loss = cnt = 0
        for start_idx in range(0, H * W, args.batch_size):
            end_idx = min(H * W, start_idx + args.batch_size)

            batch_indices = indices[start_idx:end_idx].to(device)
            batch_coords = coords[batch_indices, ...].unsqueeze(0)

            pixel_vals_preds = model(batch_coords)

            loss = ((pixel_vals_preds - gt[batch_indices, :]) ** 2).mean()
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                reconst_arr[batch_indices, :] = pixel_vals_preds.squeeze(0)

            cnt += 1

        # # no batch training -> only for comparison
        # pixel_vals_preds = model(coords.unsqueeze(0))
        # loss = ((pixel_vals_preds - gt) ** 2).mean()
        # train_loss += loss.item()

        # model.zero_grad()
        # loss.backward()
        # optim.step()

        # with torch.no_grad():
        #     reconst_arr = pixel_vals_preds.squeeze(0)

        # cnt += 1
        # # no batch training -> only for comparison

        # evaluation
        with torch.no_grad():
            reconst_arr = reconst_arr.detach().cpu().numpy()
            psnr_val = utils.psnr(gt.detach().cpu().numpy(), reconst_arr)
            psnr_vals.append(psnr_val)

            prog_bar.set_description(f"PSNR: {psnr_val:.1f} dB")
            prog_bar.refresh()

            if wandb_xp:
                wandb_xp.log({"train loss": train_loss / cnt, "psnr": psnr_val})

        lr_sched.step()

        if args.live:
            cv2.imshow("Reconstruction", reconst_arr.reshape(W, H, img_dim))
            cv2.waitKey(1)

    np.save(
        os.path.join(
            os.path.join(os.getenv("RESULTS_SAVE_PATH", "."), "reconst"),
            f"{args.nonlinearity}_psnr_vals.npy",
        ),
        psnr_vals,
    )

    return model, psnr_val, reconst_arr.reshape(W, H, img_dim)




psnr_vals = []

code = 'main'

# Read image and scale. A scale of 0.5 for parrot image ensures that it
# fits in a 12GB GPU
input_image = plt.imread(args.input_image)
img_size = (128,128)
img = cv2.resize(input_image, img_size, None, interpolation=cv2.INTER_AREA)

print(f'img_size: {img_size}')

for i in range(N_TIMES):
    if code == 'main':
        _,best_psnr,_ = train_main(args, img)
    else:
        best_psnr = train_cvpr(args, img)

    print(best_psnr)
    psnr_vals.append(float(best_psnr))

print(psnr_vals)

import json

imname = args.input_image.strip('.').split('/')[-1].split('.')[0]
print(imname)
f_name = f"{code}_{args.nonlinearity}_{imname}_e{args.epochs}_b{args.batch_size}_s{img_size[0]}__psnr_vals.json"
print(f_name)
with open(f_name, "w") as f:
    json.dump({"config": config_str, "psnr_vals": psnr_vals}, f, indent=3)

