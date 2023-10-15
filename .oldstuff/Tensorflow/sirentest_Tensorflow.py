import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import time
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from PIL import Image
print("numpy version : {}".format(np.__version__))
print("tensorflow version : {}".format(tf.__version__))


def make_coordiante(shape, min_r, max_r):
    x_coordinates = np.linspace(min_r, max_r, shape[0])
    y_coordinates = np.linspace(min_r, max_r, shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    Coordinates = np.stack([x_coordinates, y_coordinates]).T
    return Coordinates


class SineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.0):
        super(SineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(
            self.units,
            use_bias=bias,
            kernel_initializer=self.init_weights(),
            input_shape=(self.in_features,),
        )

    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(
                minval=-1 / self.in_features, maxval=1 / self.in_features
            )
        else:
            return tf.keras.initializers.RandomUniform(
                minval=-np.sqrt(6.0 / self.in_features) / self.omega_0,
                maxval=np.sqrt(6.0 / self.in_features) / self.omega_0,
            )

    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = tf.sin(self.omega_0 * befor_activation)
        # after_activation = befor_activation
        return after_activation


class Siren(tf.keras.Model):
    def __init__(
        self,
        in_features,
        units,
        out_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super(Siren, self).__init__()
        self.in_features = in_features
        self.units = units
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = []
        self.net.append(
            SineLayer(
                self.in_features,
                self.units[0],
                is_first=True,
                omega_0=self.first_omega_0,
            )
        )

        for counter in range(1, len(units)):
            self.net.append(
                SineLayer(
                    self.units[counter - 1],
                    self.units[counter],
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.outermost_linear:
            self.net.append(
                tf.keras.layers.Dense(
                    self.out_features, kernel_initializer=self.init_weights()
                )
            )
        else:
            self.net.append(
                SineLayer(
                    self.units[counter],
                    self.out_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

    def init_weights(self):
        return tf.keras.initializers.RandomUniform(
            minval=-np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
            maxval=np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
        )

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.net:
            x = layer(x)
        return x


# Image = Image.fromarray(skimage.data.camera())
Image = skimage.data.camera()
# plt.imshow(Image)

Image = Image.astype(np.float32)
Image = Image / 255

tf.keras.backend.clear_session()

batch_size = 512 * 512


omega_0 = 30.0
hidden_units = 256

siren_model = Siren(
    2,
    [hidden_units, hidden_units, hidden_units, hidden_units],
    1,
    outermost_linear=True,
    first_omega_0=omega_0,
)
_ = siren_model(np.random.rand(1, 2))

siren_model.summary()

epochs = 10
Coordinates = make_coordiante(Image.shape, -1, 1)
RGB_values = Image.flatten()

dataset = tf.data.Dataset.from_tensor_slices((Coordinates, RGB_values))
dataset = dataset.batch(batch_size)
# dataset = dataset.batch(batch_size).shuffle(1)

# optimizer = tf.keras.optimizers.Adam(1e-4)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08
)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

siren_model.compile(optimizer=optimizer, loss=loss)

history = siren_model.fit(dataset, epochs=epochs, verbose=1)

recim = siren_model.predict(Coordinates).reshape(Image.shape)
model_psnr = 20 * np.log10(1.0 / np.sqrt(history.history["loss"]))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(Image.squeeze(), cmap="gray")
axes[1].imshow(recim.squeeze(), cmap="gray")
axes[2].plot(model_psnr)
axes[2].grid()
axes[0].title.set_text("Original")
axes[1].title.set_text("Reconstructed Tensorflow")
axes[2].title.set_text("PSNR Tensorflow")
fig.suptitle("Tensorflow", fontsize=16)
plt.show()

final_psnr = 10 * np.log10(
    1 / np.mean((recim.reshape(-1, 1) - RGB_values.reshape(-1, 1)) ** 2)
)
print(final_psnr)
with open(f"data_Siren_Tensorflow_{epochs}.data", "wb") as f:
    pickle.dump((model_psnr, recim), f)
