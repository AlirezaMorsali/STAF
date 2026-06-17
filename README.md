# A Unified Theory of Sinusoidal Activation Families for Implicit Neural Representations

Official code for **STAF**, published in **Transactions on Machine Learning Research (TMLR), June 2026**.

[Project page](https://alirezamorsali.github.io/staf/) | [OpenReview](https://openreview.net/forum?id=ZDmBPYptbL) | [Paper PDF](https://openreview.net/pdf?id=ZDmBPYptbL)

**Authors:** Alireza Morsali, MohammadJavad Vaez, Mohammadhossein Soltani, Amirhossein Kazerouni, Babak Taati, Morteza Mohammad-Noori

**Affiliations:** McGill University, The University of Melbourne, MACSYS, University of Toronto, Vector Institute, University Health Network, University of Tehran

## Overview

Implicit Neural Representations (INRs) model continuous signals with compact neural networks and are widely used in vision, graphics, and signal processing. This work studies a broad family of sinusoidal activations for INRs and instantiates that view with **STAF**, a trainable Fourier-like activation whose amplitudes, frequencies, and phases are learned directly.

The paper develops a unified theoretical and practical framework for sinusoidal activation families, including a Kronecker-equivalence result, an NTK-based capacity and convergence analysis, and an initialization scheme with unit-variance post-activations. Empirically, STAF is competitive and often stronger on image fitting, audio reconstruction, shape representation, inverse problems, and NeRF.

<p align="center">
  <img src="./docs/staf_comparison_graphs.png" width="850" alt="STAF comparison graphs">
</p>

## Key Contributions

- Unifies SIREN and related multi-sinusoid INR activations under a single sinusoidal activation family.
- Shows how trainable sinusoidal parameters expand effective frequency support and reshape optimization behavior.
- Provides a practical initialization and parameter-sharing recipe that works across INR tasks.
- Evaluates STAF on images, audio, 3D shapes, denoising, super-resolution, and NeRF.

## Repository Contents

- `modules/staf.py`: STAF activation and model components.
- `modules/`: INR baselines and shared utilities used by the experiments.
- `train_image.ipynb`: Image representation experiments.
- `train_audio.ipynb`: Audio reconstruction experiments.
- `train_sdf.ipynb`: 3D shape representation experiments.
- `train_denoising.ipynb`: Image denoising experiments.
- `train_sr.ipynb`: Image super-resolution experiments.
- `docs/`: Figures used in this README.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The notebooks expect the paper data to be available under a `data/` directory at the repository root.

Download the data used in the paper from [Google Drive](https://drive.google.com/file/d/1AXr64xXE_oQMIWBpgzZuh4ZtWEVyupLM/view?usp=drive_link), unzip it, and copy the extracted files into `data/`.

## Reproducing Experiments

### Image Representation

Run `train_image.ipynb` to reproduce the image fitting experiments. The project page highlights sharper reconstructions and stronger late-stage convergence for STAF compared with several INR baselines.

### Audio Reconstruction

Run `train_audio.ipynb` to reproduce the audio reconstruction experiments. The paper reports the highest PSNR and lowest reconstruction error for STAF on the featured Bach cello example.

### Shape Representation

Run `train_sdf.ipynb` to reproduce the 3D shape representation experiments. The data release includes occupancy volumes for Lucy, Thai, Armadillo, and Dragon.

The shape notebook writes `.dae` mesh files, which can be opened with tools such as [MeshLab](https://www.meshlab.net/).

### Image Denoising

Run `train_denoising.ipynb` to reproduce the denoising experiments. These evaluate STAF on recovering fine image structure under severe corruption.

### Image Super-resolution

Run `train_sr.ipynb` to reproduce the super-resolution experiments, including the 4x setting shown on the project page.

### NTK Analysis

The empirical NTK eigenfunctions and eigenvalue spectrum shown on the project page can be reproduced with `figure_5.py` from the `inr_dictionaries` code used by the paper.

## Citation

If you use this code or build on the paper, please cite:

```bibtex
@article{
morsali2026a,
title={A Unified Theory of Sinusoidal Activation Families for Implicit Neural Representations},
author={Alireza Morsali and MohammadJavad Vaez and Mohammad Hossein Soltani and Amirhossein Kazerouni and Babak Taati and Morteza Mohammad-Noori},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=ZDmBPYptbL},
note={}
}
```

## License

This repository is released under the license provided in [LICENSE](LICENSE).
