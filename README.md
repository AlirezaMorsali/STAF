# STAF: Sinusoidal Trainable Activation Functions for Implicit Neural Representation

---

### [Project Page](#) | [Paper](https://arxiv.org/abs/2502.00869) | [Data](#)

**STAF** is a novel approach that enhances Implicit Neural Representations (INRs) by introducing trainable sinusoidal activation functions. Specifically, STAF dynamically modulates its frequency components, enabling networks to adaptively learn and represent complex signals with higher precision and efficiency. It excels in signal representation, handling various tasks such as image, shape, and audio reconstructions, and tackles complex challenges like spectral bias and inverse problems, outperforming state-of-the-art methods in accuracy and reconstruction fidelity.

> [*Alireza Morsali*](), [*MohammadJavad Vaez*](https://mjvaez.github.io), [*Hossein Soltani*](https://neotod.github.io/), [*Amirhossein Kazerouni*](https://amirhossein-kz.github.io/), [*Babak Taati*](), [*Morteza Mohammad-Noori*]()
>

<br>

<p align="center">
  <img src="./docs/staf_comparison_graphs.png" width="850">
</p>


## Get started

### Data
You can download the data utilized in the paper from this  [link]().

### Requirements
Install the requirements with:
```bash
pip install -r requirements.txt
```


### Image Representation
The image experiment can be reproduced by running the `train_image.ipynb` notebook.

### Audio Representation
The audio experiment can be reproduced by running the `train_audio.ipynb` notebook.

### Shape Representation
The shape experiment can be reproduced by running the `train_sdf.ipynb` notebook. For your convenience, we have included the occupancy volume of Lucy with regular sampling in 512x512x512 cubes in the data file. 

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
>   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
> </picture><br>
> To test the model with custom input data, you can run the <code>preprocess_sdf.ipynb</code> notebook, which will generate a pre-processed <code>.npy</code> file for your desired input.
>
> <br>
>  The output is a <code>.dae</code> file that can be visualized using software such as Meshlab (a cross-platform visualizer and editor for 3D models).

### Image Denoising
The denoising experiment can be reproduced by running the `train_denoising.ipynb` notebook.

### Image Super-resolution
The super-resolution experiment can be reproduced by running the `train_sr.ipynb` notebook.

### NTK 
The NTK eigenfunctions and eigenvalues can be reproduced by running the `figure_5.py` of [inr_dictionaries](https://github.com/neotod/inr_dictionaries/).


## Acknowledgement
We thank the authors of [INCODE](https://github.com/xmindflow/INCODE), [inr_dictionaries](https://github.com/gortizji/inr_dictionaries) for their code repositories.


## Citation
```bibtex
@article{morsali2025staf,
  title={STAF: Sinusoidal Trainable Activation Functions for Implicit Neural Representation},
  author={Morsali, Alireza and Vaez, MohammadJavad and Soltani, Hossein and Kazerouni, Amirhossein and Taati, Babak and Mohammad-Noori, Morteza},
  journal={arXiv preprint arXiv:2502.00869},
  year={2025}
}
```