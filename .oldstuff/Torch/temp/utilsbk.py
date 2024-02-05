import numpy as np

def custumparser(parser):
        def is_interactive():
                import __main__ as main
                return not hasattr(main, '__file__')

        if is_interactive():
                par = parser.parse_args([])
        else:
                par = parser.parse_args()
        return par


def make_coordiante(shape):
    x_coordinates = np.linspace(-1, +1, shape[0])
    y_coordinates = np.linspace(-1, +1, shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    Coordinates = np.stack([x_coordinates, y_coordinates]).T
    return Coordinates


def load_image():
        Image = skimage.data.camera()
        plt.imshow(Image, cmap='gray')
        Image = Image.astype(np.float32)
        Image = Image / 255
