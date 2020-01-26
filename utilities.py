import numpy as np


def upsampling(sig, up):
    sig = np.atleast_2d(sig[:])
    sig_new = np.zeros((sig.shape[0], sig.shape[1] * up), dtype=sig.dtype)

    for index, row in enumerate(sig_new):
        row[::up] = sig[index]

    return sig_new


def scatterplot(samples, sps=1):
    import matplotlib.pyplot as plt

    fignumber = samples.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=fignumber)
    for ith, ax in enumerate(axes):
        ax.scatter(samples[ith, ::sps].real, samples[ith, ::sps].imag, s=1, c='b')
        ax.set_aspect('equal', 'box')

        ax.set_xlim([samples[ith, ::sps].real.min() - 0.1, samples[ith, ::sps].real.max() + 0.1])
        ax.set_ylim(
            [samples[ith, ::sps].imag.min() - 0.1, samples[ith, ::sps].imag.max() + 0.1])

    plt.tight_layout()
    plt.show()


class Osa(object):

    def __init__(self, resolution):
        self.resolution = resolution
