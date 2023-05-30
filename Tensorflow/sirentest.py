import argparse
import skimage
from matplotlib import pyplot as plt
from tensorflow import keras
from lib import nrnets, utils
import os
import tensorflow as tf
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser(description="Parametric Neural Representaiton")
parser.add_argument(
    "-n",
    "--ExperimentName",
    dest="expname",
    type=str,
    default="Siren Camera",
    help="Experiment name for storing the results",
)
parser.add_argument(
    "-ad",
    "--ImageAddress",
    dest="addr",
    type=str,
    default=None,
    help="Local address (or URL) of the target image, if None,\
                    ski-image samples",
)
parser.add_argument(
    "-sk",
    "--SkImage",
    dest="skim",
    type=str,
    default="camera",
    help="Image Sample from Scikit-image",
)
parser.add_argument(
    "-ep",
    "--Epochs",
    dest="ep",
    type=int,
    default=500,
    help="Number of Epochs",
)

# parser.add_argument('-L', '--Lpath', dest='L', type=int,  default=20,
#                     help='Number of paths')
# parser.add_argument('-Ntx', '--NumTranAnt', dest='Ntx', type=int,  default=8,
#                     help='Number of transmit antennas')
# parser.add_argument('-Nrx', '--NumRecvAnt', type=int, dest='Nrx', default=128,
#                     help='Number of receive antennas')


par = utils.custumparser(parser)
par.ep = 10
# par.addr = "Data/lena512color.tiff"

# dataset, imshape, coordinates, img = utils.image_dataloader(par, False)

dataset, imshape, coordinates, img = utils.image_dataloader_db()


# plt.imshow(dataset)
# siren_model = nrnets.get_siren(
#     in_features=2,
#     out_features=imshape[2],
#     units=[256, 256, 256, 256],
#     outermost_linear=True,
#     first_omega_0=30.0,
#     hidden_omega_0=30.0,
# )

hidden_units = 256
siren_model = nrnets.Siren(2, [hidden_units, hidden_units, hidden_units, hidden_units], 1, outermost_linear=True)
_ = siren_model(np.random.rand(1, 2))


keras.utils.plot_model(siren_model, "_Model.png", show_shapes=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
siren_model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

history = siren_model.fit(dataset, epochs=par.ep)
siren_loss = np.array(history.history['loss'])

plt.plot(siren_loss)

siren_psnr = 10 * np.log10(1.0 / (siren_loss))
plt.plot(siren_psnr)

recim = siren_model.predict(coordinates).reshape(imshape)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img.squeeze(), cmap='gray')
axes[1].imshow(recim.squeeze(), cmap='gray')
axes[2].plot(siren_psnr)
axes[2].grid()
axes[0].title.set_text('Original')
axes[1].title.set_text('Reconstructed Tensorflow')
axes[2].title.set_text('PSNR Tensorflow')
fig.suptitle('Clean Code Tensorflow', fontsize=16)
plt.show()


# fig = plt.figure()
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.imshow(img.squeeze(), cmap='gray')
# ax1.set_title('Original')
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.imshow(recim.squeeze(), cmap='gray')
# ax2.set_title('Reconstructed')
# plt.show()
with open(f"data_{par.expname}.data", "wb") as f:
    pickle.dump(siren_psnr, f)


# with open("data_shuffeled.data", "rb") as f:
#     psnr = pickle.load(f)
