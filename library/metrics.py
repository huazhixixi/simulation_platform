import numpy as np


def symbol2bin(symbol, width):
    symbol = np.atleast_2d(symbol)[0]
    biterror = np.empty((len(symbol), width), dtype=np.bool)
    new_symbol = symbol.copy()
    new_symbol.shape = -1, 1
    for i in np.arange(width):
        biterror[:, i] = ((new_symbol >> i) & 1)[:, 0]

    return biterror


def calc_ber(rx_symbols, tx_symbols, qam_order):
    nbit = int(np.log2(qam_order))
    if 2 ** nbit <= np.max(rx_symbols) or 2 ** nbit <= np.max(tx_symbols):
        raise Exception("Qam order is wrong")

    rx_symbols = np.atleast_2d(rx_symbols)[0]
    tx_symbols = np.atleast_2d(tx_symbols)[0]
    mask = rx_symbols != tx_symbols

    error1 = symbol2bin(rx_symbols[mask], nbit)
    error2 = symbol2bin(tx_symbols[mask], nbit)
    error_num = np.sum(np.logical_xor(error1, error2))
    return error_num / (len(rx_symbols) * nbit)