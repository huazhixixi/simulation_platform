import numpy as np


def rcos_freq(f, beta, T, is_on_cuda=False):
    """Frequency response of a raised cosine filter with a given roll-off factor and width """
    if is_on_cuda:
        import cupy as np
    else:
        import numpy as np
    rc = np.zeros(f.shape[0], dtype=f.dtype)
    rc[np.where(np.abs(f) <= (1 - beta) / (2 * T))] = T
    idx = np.where((np.abs(f) > (1 - beta) / (2 * T)) & (np.abs(f) <= (
            1 + beta) / (2 * T)))
    rc[idx] = T / 2 * (1 + np.cos(np.pi * T / beta *
                                  (np.abs(f[idx]) - (1 - beta) /
                                   (2 * T))))
    return rc


def rrcos_freq(f, beta, T, is_on_cuda=False):
    """Frequency transfer function of the square-root-raised cosine filter with a given roll-off factor and time width/sampling period after _[1]
    Parameters
    ----------
    f   : array_like
        frequency vector
    beta : float
        roll-off factor needs to be between 0 and 1 (0 corresponds to a sinc pulse, square spectrum)
    T   : float
        symbol period
    Returns
    -------
    y   : array_like
       filter response
    References
    ----------
    ..[1] B.P. Lathi, Z. Ding Modern Digital and Analog Communication Systems
    """
    return np.sqrt(rcos_freq(f, beta, T, is_on_cuda))


def rrcos_pulseshaping_freq(sig, fs, T, beta, is_on_cuda=False):
    """
    Root-raised cosine filter in the spectral domain by multiplying the fft of the signal with the
    frequency response of the rrcos filter.
    Parameters
    ----------
    sig    : array_like
        input time distribution of the signal
    fs    : float
        sampling frequency of the signal
    T     : float
        width of the filter (typically this is the symbol period)
    beta  : float
        filter roll-off factor needs to be in range [0, 1]
    Returns
    -------
    sign_out : array_like
        filtered signal in time domain
    """
    if is_on_cuda:
        import cupy as np
    else:
        import numpy as np

    f = np.fft.fftfreq(sig.shape[0]) * fs
    nyq_fil = rrcos_freq(f, beta, T, is_on_cuda)
    nyq_fil /= nyq_fil.max()
    sig_f = np.fft.fft(sig)
    sig_out = np.fft.ifft(sig_f * nyq_fil)
    return sig_out
