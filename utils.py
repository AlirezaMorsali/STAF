#!/usr/bin/env python

"""
Miscellaneous utilities that are helpful but cannot be clubbed into other modules.
"""

# Scientific computing
import numpy as np


def normalize(x, full_normalize=False):
    """
    Normalize input to lie between 0, 1.

    Inputs:
        x: Input signal
        full_normalize: If True, normalize such that minimum is 0 and
            maximum is 1. Else, normalize such that maximum is 1 alone.

    Outputs:
        xnormalized: Normalized x.
    """

    if x.sum() == 0:
        return x

    xmax = x.max()

    if full_normalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin) / (xmax - xmin)

    return xnormalized


def psnr(x, xhat):
    """Compute Peak Signal to Noise Ratio in dB

    Inputs:
        x: Ground truth signal
        xhat: Reconstructed signal

    Outputs:
        snrval: PSNR in dB
    """
    err = x - xhat
    denom = np.mean(pow(err, 2))

    snrval = 10 * np.log10(np.max(x) / denom)

    return snrval


def count_parameters(model):
    """Count the number of paramaters.
    Inputs:
        model: Model instance

    Outputs:
        Number of parameters of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
