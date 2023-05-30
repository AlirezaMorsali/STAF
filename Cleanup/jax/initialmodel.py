#!/usr/bin/env jupyter

from typing import Sequence

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


image_path = "Data/lena512color.tiff"
image = plt.imread(image_path)

c = [image.shape[0] // 2, image.shape[1] // 2]
r = 256
mage = image[c[0] - r:c[0] + r, c[1] - r:c[1] + r]
image = image[::2, ::2] / 255.0
plt.imshow(image)

model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
output = model.apply(variables, batch)
