#!/usr/bin/env jupyter

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy
import skimage
import numpy as np
from numpy import pi



def dct2(a):
    return scipy.fft.dct(scipy.fft.dct(a, axis=0, norm="ortho"),
                         axis=1,
                         norm="ortho")


def idct2(a):
    return scipy.fft.idct(scipy.fft.idct(a, axis=0, norm="ortho"),
                          axis=1,
                          norm="ortho")


img = skimage.img_as_float64(skimage.data.camera())
dct = dct2(img)
dct_shifted = scipy.fft.fftshift(dct2(img))
rec = idct2(dct)



plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Original Image")

(row, col) = dct.shape
X = np.linspace(-5, 5, row)
Y = np.linspace(-5, 5, col)
X, Y = np.meshgrid(X, Y)
# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, dct_shifted, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


plt.figure()
plt.imshow(rec, cmap="gray")
plt.title("Reconstructed image form DCT")
plt.show()
