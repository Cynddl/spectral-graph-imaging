from spectral import *
from scipy import misc


def cumulative_spectrum(f_signal, eigenvectors, shape):
    """
    Given the eigenvectors of the graph laplacian from an image,
    returns an animation of the inverse fourier transformation of the signal.
    """
    recomposed_signal = np.zeros(len(f_signal))
    for i in range(len(f_signal)):
        recomposed_signal += f_signal[i] * eigenvectors[:, i]
        yield recomposed_signal.reshape(shape)


def write_cumulative_spectrum(f_signal, eigenvectors, shape, out_dir=''):
    out_path = out_dir + 'animation_'

    for i, image_signal in enumerate(cumulative_spectrum(f_signal, eigenvectors, shape)):
        misc.imsave(out_path + str(i) + '.png', image_signal)
