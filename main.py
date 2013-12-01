from spectral import *
from scipy import misc


def main():
    # Lena picture
    l = misc.lena()
    l = misc.imresize(l, (64, 64))
    l = l.astype(float)
    l *= 1. / l.max()

    # # Black and white sample
    # l = np.array([[1 if i > 20 and i < 44 and j > 20 and j < 44 else 0 for j in range(64)] for i in range(64)])

    # # With random noise
    # l = np.array([[np.random.random() * 0.1 + (1 if i > 20 and i < 44 and j > 20 and j < 44 else 0) for j in range(64)] for i in range(64)])

    out_dir = 'out/box_noise/'

    graph = ImageGraph(l)

    print("Plotting the raw image.")
    plot_signal(graph.img, name=out_dir + 'raw.png')

    print("Computing all the weights of the graph...")
    graph.apply_weight(sigma=0.1)
    print("Computing spectral values of the graph...")
    eigenvalues, eigenvectors = graph.compute_spectral()
    # eigenvalues, eigenvectors = graph.compute_sparse_spectral(k=200)

    # plt.plot(eigenvalues)
    # plt.show()

    plot_signal(eigenvectors[:, 0], shape=graph.shape, name=out_dir + 'vect_0.png')
    plot_signal(eigenvectors[:, 1], shape=graph.shape, name=out_dir + 'vect_1_fiedler.png')
    plot_signal(eigenvectors[:, 2], shape=graph.shape, name=out_dir + 'vect_2.png')
    plot_signal(eigenvectors[:, 3], shape=graph.shape, name=out_dir + 'vect_3.png')
    plot_signal(eigenvectors[:, -1], shape=graph.shape, name=out_dir + 'vect_minus_1.png')

    print("Computing inverse fourier transformation")
    fourier_signal = np.dot(graph.img.flatten(), eigenvectors)

    inv_fourier = inverse_fourier(fourier_signal, eigenvectors)
    inv_fourier_filtered = inverse_fourier_filter(fourier_signal, eigenvectors, eigenvalues, gamma=10)

    plot_signal(inv_fourier, shape=graph.shape, name=out_dir + 'inverse_fourier.png')
    plot_signal(inv_fourier_filtered, shape=graph.shape, name=out_dir + 'inverse_fourier_filtered.png')


if __name__ == '__main__':
    main()
