import numpy as np
import skimage
from skimage import io
import tensorflow as tf
import matplotlib.pylab as plt


def custumparser(parser):
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')

    if is_interactive():
        par = parser.parse_args([])
    else:
        par = parser.parse_args()
    return par


def image_dataloader(par, showimage=True):
    if par.addr:
        img = skimage.util.img_as_float(io.imread(par.addr))
    else:
        temp = f"skimage.data.{par.skim}()"
        img = skimage.util.img_as_float(eval(temp))
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    if showimage:
        if img.shape[2] == 1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
    x_coordinates = np.linspace(-1, +1, img.shape[0])
    y_coordinates = np.linspace(-1, +1, img.shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    # rgb_values = img.transpose(2, 0, 1).reshape(img.shape[2], -1).T
    rgb_values = img.flatten()
    dataset = tf.data.Dataset.from_tensor_slices((coordinates, rgb_values))
    dataset = dataset.shuffle(img.shape[0]*img.shape[1]).batch(
        img.shape[0]*img.shape[1])
    return (dataset, img.shape, coordinates, img)


def image_dataloader_db():
    def make_coordiante(shape, min_r, max_r):
        x_coordinates = np.linspace(min_r, max_r, shape[0])
        y_coordinates = np.linspace(min_r, max_r, shape[1])
        x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()
        Coordinates = np.stack([x_coordinates, y_coordinates]).T
        return Coordinates
    Image = skimage.data.camera()
    # plt.imshow(Image)
    Image = Image.astype(np.float32)
    Image = Image / 255
    Coordinates = make_coordiante(Image.shape, -1, 1)
    RGB_values = Image.flatten()
    dataset = tf.data.Dataset.from_tensor_slices((Coordinates, RGB_values))
    dataset = dataset.batch(512*512)
    return (dataset, Image.shape, Coordinates, Image)
