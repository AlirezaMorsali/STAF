import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import matplotlib.font_manager as font_manager
import skimage
import math
import time
from math import log10, sqrt
import imageio
import os
from lib import nrnets
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("numpy version : {}".format(np.__version__))
print("tensorflow version : {}".format(tf.__version__))


def cm2inch(*tupl):
    inch = 2.54

    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        if epoch % 100 == 0:
            lr = lr * tf.math.exp(-0.05)

        return lr


epochs = 15000
step_show = 100

Image = skimage.data.camera()
plt.imshow(Image, cmap='gray')
Image = Image.astype(np.float32)
Image = Image / 255

Coordinates = nrnets.make_coordiante(Image.shape)
RGB_values = Image.flatten()
n_fourier_features = 10
scale = 10

B = np.random.randn(2, n_fourier_features)
B = scale * B

Coordinates = Coordinates @ B
Coordinates = np.concatenate([np.sin(2 * math.pi * Coordinates),
                              np.cos(2 * math.pi * Coordinates)], axis=-1)

batch_size = 16 * 1024

if len(Image.shape) == 3:
    output_channel = 3
else:
    output_channel = 1


n_fourier_features = 10
scale = 10

B = np.random.randn(2, n_fourier_features)
B = scale * B

save_result_siren = nrnets.CustomSaver(Image.shape, step_show, batch_size)
save_result_param = nrnets.CustomSaver(Image.shape, step_show, batch_size)
save_result_relu = nrnets.CustomSaver(Image.shape, step_show, batch_size)
save_result_fourier = nrnets.CustomSaver(Image.shape, step_show, batch_size,
                                         is_fourier=True, B=B)

omega_0 = 30
hidden_units = 256
hidden_initializers = tf.keras.initializers.RandomUniform(minval=-np.sqrt(
    6/hidden_units)/omega_0, maxval=np.sqrt(6/hidden_units)/omega_0)

# SIREN Model
X = tf.keras.layers.Input(shape=(2,))
x1 = nrnets.SineLayer(2, hidden_units, is_first=True, omega_0=30)(X)
x2 = nrnets.SineLayer(hidden_units, hidden_units, is_first=False)(x1)
x3 = nrnets.SineLayer(hidden_units, hidden_units, is_first=False)(x2)
x4 = nrnets.SineLayer(hidden_units, hidden_units, is_first=False)(x3)

x = tf.keras.layers.Add()([x1, x2, x3, x4])

Y = tf.keras.layers.Dense(output_channel,
                          kernel_initializer=hidden_initializers)(x)
siren_model = tf.keras.models.Model(X, Y)

# Parametric Model
X = tf.keras.layers.Input(shape=(2,))
x1 = nrnets.ParaSineLayer(2, hidden_units, is_first=True, omega_0=30)(X)
x2 = nrnets.ParaSineLayer(hidden_units, hidden_units, is_first=False)(x1)
x3 = nrnets.ParaSineLayer(hidden_units, hidden_units, is_first=False)(x2)
x4 = nrnets.ParaSineLayer(hidden_units, hidden_units, is_first=False)(x3)
x = tf.keras.layers.Add()([x1, x2, x3, x4])
Y = tf.keras.layers.Dense(output_channel,
                          kernel_initializer=hidden_initializers)(x)
parametric_model = tf.keras.models.Model(X, Y)

# Fourier Model
X = tf.keras.layers.Input(shape=(2 * n_fourier_features,))
x1 = tf.keras.layers.Dense(hidden_units, activation='relu')(X)
x2 = tf.keras.layers.Dense(hidden_units, activation='relu')(x1)
x3 = tf.keras.layers.Dense(hidden_units, activation='relu')(x2)
x4 = tf.keras.layers.Dense(hidden_units, activation='relu')(x3)
x = tf.keras.layers.Add()([x1, x2, x3, x4])
Y = tf.keras.layers.Dense(output_channel)(x)
fourier_model = tf.keras.models.Model(X, Y)

# Relu Model
X = tf.keras.layers.Input(shape=(2,))
x1 = tf.keras.layers.Dense(hidden_units, activation='relu')(X)
x2 = tf.keras.layers.Dense(hidden_units, activation='relu')(x1)
x3 = tf.keras.layers.Dense(hidden_units, activation='relu')(x2)
x4 = tf.keras.layers.Dense(hidden_units, activation='relu')(x3)
x = tf.keras.layers.Add()([x1, x2, x3, x4])
Y = tf.keras.layers.Dense(output_channel)(x)
relu_model = tf.keras.models.Model(X, Y)

siren_model.summary()
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

models = [relu_model,
          fourier_model,
          siren_model,
          parametric_model]

callbacks = [save_result_relu,
             save_result_fourier,
             save_result_siren,
             save_result_param]

model_name = [
              "Relu",
              "Fourier Feature",
              "Siren",
              "Parametric"
]

with_fourier = [False,
                True,
                False,
                False]


History = []


for counter in range(len(models)):
    Coordinates = nrnets.make_coordiante(Image.shape)

    if with_fourier[counter]:
        Coordinates = Coordinates @ B
        Coordinates = np.concatenate([np.sin(2 * math.pi * Coordinates),
                                      np.cos(2 * math.pi * Coordinates)],
                                     axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((Coordinates, RGB_values))
    dataset = dataset.shuffle(len(Coordinates)).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.MeanSquaredError()

    models[counter].compile(optimizer=optimizer, loss=loss)

    print(20*"*", model_name[counter], 20*"*")
    history = models[counter].fit(dataset, epochs=epochs,
                                  callbacks=[callback, callbacks[counter]])
    History.append(history.history['loss'])

plt.plot(np.stack(History).T)

# Video = []
# # model_psnr = []

# for save_result in callbacks:
#     print(save_result.images)

#     Video.append(np.stack(save_result.images))
# # psnr_buff = []
# # for image in save_result.images:
# # psnr_buff.append(PSNR(Image, image))
# # model_psnr.append(psnr_buff)
# # model_psnr = np.stack(model_psnr)
# Video = np.concatenate(Video, axis=2)
model_psnr = 20 * np.log10(1.0 / np.sqrt(np.stack(History).T)).T


# if output_channel == 3:
#     images_gt = np.tile(np.expand_dims(Image, axis=0),
#                         (int(epochs / step_show) + 1, 1, 1, 1))
# else:
#     images_gt = np.tile(np.expand_dims(Image, axis=0),
#                         (int(epochs / step_show) + 1, 1, 1))
# Video = np.concatenate([images_gt, Video], axis=2)
# Video = (255 * np.clip(Video, 0, 1)).astype(np.uint8)
# f = os.path.join('training_convergence.mp4')
# imageio.mimwrite(f, Video, fps=3)

# text_font = {'fontname': 'Serif', 'size': '20'}

# plt.figure(figsize=(25, 5))
# plt.imshow(Video[3], cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.text(1 * 256 - 100, -20, 'Original', **text_font)
# plt.text(3 * 256 - 50, -20, 'Relu', **text_font)
# plt.text(5 * 256 - 120, -20, 'Fourier Features', **text_font)
# plt.text(7 * 256 - 50, -20, 'Siren', **text_font)
# plt.text(9 * 256 - 140, -20, 'Parametric(Ours)', **text_font)

# plt.savefig('Compresion_on_Epoch20.pdf', format='pdf', dpi=600,
#             bbox_inches='tight', pad_inches=0)


mpl.rcParams['pdf.fonttype'] = 3
mpl.rcParams['ps.fonttype'] = 3
mpl.rcParams['font.family'] = 'Serif'


mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.labelweight'] = "bold"


mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6


mpl.rcParams['legend.fontsize'] = 5

fig = plt.figure(figsize=cm2inch(9.4, 6), dpi=600)
ax = plt.axes()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


legend_font = font_manager.FontProperties(family='Serif')


plt.plot(model_psnr.T, linewidth=0.75)

# plt.ylim([0, 50])

# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.xaxis.set_major_locator(MultipleLocator(5))
# ax.yaxis.set_major_locator(MultipleLocator(10))
# ax.yaxis.set_minor_locator(MultipleLocator(2))

plt.xlabel('Epoch')
plt.ylabel('Psnr')
ax.legend(['Relu', 'Fourier Features', 'Siren', 'Parametric (Ours)'],
          loc='upper left',
          bbox_to_anchor=(0, 1),
          handletextpad=0.2,
          frameon=False)

plt.savefig('PSNR_Compresion.pdf', format='pdf', dpi=600, bbox_inches='tight',
            pad_inches=0)
