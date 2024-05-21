#!/usr/bin/env python3

import os
import skimage
from tqdm import tqdm
import time
import cv2
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from models import Parac, Wire, Siren
from modules import models
import utils


load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")


def get_args():
    """
    Get training arguemtns.

    Outputs:
        args: arguemnts object.
    """
    parser = argparse.ArgumentParser(description="Image reconstruction parameters")
    parser.add_argument(
        "-i",
        "--input_image",
        type=str,
        help="Input image name from skimage.",
        default="camera",
    )
    parser.add_argument(
        "-n",
        "--non_linearity",
        choices=["parac", "wire", "siren", "kan"],
        type=str,
        help="Name of non linearity",
        default="parac",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="Epcohs of maining", default=250
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Parac works best at 1e-4, Wire at 5e-3 to 2e-2, SIREN at 1e-3 to 2e-3.",
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

    return parser.parse_args()


def save_results(train_name, best_image, model, results_saving_path, model_saving_path):
    """
    Save the results. Including saving the reconstructed image and the trained model.
    """

    print("Saving the reconstructed image and the trained model.")
    # save model
    os.makedirs(
        model_saving_path,
        exist_ok=True,
    )
    torch.save(
        model.state_dict(),
        os.path.join(
            model_saving_path,
            f"{train_name}.pth",
        ),
    )

    # saving the results
    os.makedirs(
        results_saving_path,
        exist_ok=True,
    )
    plt.imshow(best_image)
    plt.savefig(
        os.path.join(
            results_saving_path,
            f"{train_name}.png",
        )
    )

    np.save(
        os.path.join(
            results_saving_path,
            f"{train_name}.npy",
        ),
        best_image,
    )

    if os.getenv("WANDB_LOG") in ["true", "True", True]:
        print("saving the image on WANDB")
        wandb.log(
            {
                f"image_reconst | {train_name}": [
                    wandb.Image(best_image, caption="Reconstructed image.")
                ]
            }
        )


def get_model(
    non_linearity,
    hidden_features,
    hidden_layers,
    out_features,
    first_omega_0=30,
    hidden_omega_0=30,
    scale=10,
    sidelength=512,
    fn_samples=None,
    use_nyquist=True,
):
    """
    Function to get a class instance for a given type of
    implicit neural representation

    Inputs:
        non_linearity: One of 'parac', 'wire', 'siren'.
        in_features: Number of input features. 2 for image, 3 for volume and so on.
        hidden_features: Number of features per hidden layer
        hidden_layers: Number of hidden layers
        out_features; Number of outputs features. 3 for color image, 1 for grayscale or volume and so on
        first_omega0 (30): For siren and wire only: Omega for first layer
        hidden_omega0 (30): For siren and wire only: Omega for hidden layers
        scale (10): For wire and gauss only: Scale for Gaussian window
        pos_encode (False): If True apply positional encoding
        sidelength (512): if pos_encode is true, use this for side length parameter
        fn_samples (None): Redundant parameter
        use_nyquist (True): if True, use nyquist sampling for positional encoding

    Outputs:
        Model instance
    """

    if non_linearity == "parac":
        model = Parac(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            pos_encode=False,
            sidelength=sidelength,
            fn_samples=fn_samples,
            use_nyquist=use_nyquist,
        )

    elif non_linearity == "wire":
        model = Wire(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            pos_encode=False,
            sidelength=sidelength,
            fn_samples=fn_samples,
            use_nyquist=use_nyquist,
        )
    elif non_linearity == "siren":
        model = Siren(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            pos_encode=False,
            sidelength=sidelength,
            fn_samples=fn_samples,
            use_nyquist=use_nyquist,
        )

    elif non_linearity == 'kan':
        model = models.INR(non_linearity).run(
            in_features=2,
            out_features=out_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            degree=16,
        )


    return model


def load_image_data(image_name):
    """
    Loading image from skimage.
    If the image name is correct, it will return the iamge.
    Otherwise, it'll raise an exception.

    Inputs:
        image_name: name of the image - for example 'camera"

    Outputs:
        image file from skimage.
    """

    if hasattr(skimage.data, image_name):
        input_image_get_fn = getattr(skimage.data, image_name)
        return input_image_get_fn()
    else:
        raise Exception(
            f'Image name is wrong. Skimage does not have image with the name "{image_name}".'
        )


def train(args, wandb_xp=None):
    """
    Train the model.

    Inputs:
        args: training arguments.
        wandb_xp: WANDB expriment instance.

    Outputs:
        model: trained model
        reconstructed image: The last reconstructed image.
    """

    if args.non_linearity == "wire":
        # Gabor filter constants.
        # We suggest omega0 = 4 and sigma0 = 4 for reconst, and omega0=20, sigma0=30 for image representation
        omega0 = 20  # Frequency of sinusoid
        sigma0 = 30  # Sigma of Gaussian

    else:
        omega0 = 30.0  # Frequency of sinusoid
        sigma0 = 4.0  # Sigma of Gaussian

    img = load_image_data(args.input_image)
    img = cv2.resize(img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    img = utils.normalize(img.astype(np.float32), full_normalize=True)

    H, W = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # grayscale image
        img_dim = 1
        img = img[:, :, np.newaxis]
    else:
        # rgb image
        img_dim = 3

    model = get_model(
        non_linearity=args.non_linearity,
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
            f"{args.non_linearity}_psnr_vals.npy",
        ),
        psnr_vals,
    )

    return model, reconst_arr.reshape(W, H, img_dim)


def main():
    """
    Main function for training the model on Image Reconstruction task.
    """
    wandb_xp = None
    if os.getenv("WANDB_LOG") in ["true", "True", True]:
        run_name = f'image_reconst | {args.non_linearity} | {args.input_image} | {str(time.time()).replace(".", "_")}'
        wandb_xp = wandb.init(
            name=run_name, project="pracnet", resume="allow", anonymous="allow"
        )

    args = get_args()
    model, reconstructed_image = train(args, wandb_xp)
    save_results(
        f"{args.non_linearity}_{args.input_image}",
        reconstructed_image,
        model,
        os.path.join(os.getenv("RESULTS_SAVE_PATH", "."), "reconst"),
        os.path.join(os.getenv("MODEL_SAVE_PATH", "."), "reconst"),
    )


if __name__ == "__main__":
    main()
