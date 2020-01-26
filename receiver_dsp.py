from filter_design import rrcos_pulseshaping_freq


def matched_filter(signal, roll_off):
    if signal.is_on_cuda:
        import cupy as np
    else:
        import numpy as np
    samples = np.copy(signal[:])
    for row in samples:
        row[:] = rrcos_pulseshaping_freq(row, signal.fs, 1 / signal.baudrate, roll_off, signal.is_on_cuda)
    signal[:] = samples
    return signal
