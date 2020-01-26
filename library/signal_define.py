import matplotlib.pyplot as plt
import numpy as np

from .filter_design import rrcos_pulseshaping_freq
from .utilities import upsampling


class Signal(object):

    def __init__(self, qam_order, baudrate, sps, sps_in_fiber, symbol_length, pol_number):
        '''
            qam_order
            message 2d-array
            all 2d-array

            baudrate:hz

        '''
        self.qam_order = qam_order
        self.message = None
        self.baudrate = baudrate
        self.sps = sps
        self.sps_in_fiber = sps_in_fiber
        self.ds = None
        self.ds_in_fiber = None
        self.symbol = None
        self.freq = None
        self.symbol_length = symbol_length
        self.pol_number = pol_number
        self.__constl = None
        self.is_on_cuda = False

    @property
    def fs_in_fiber(self):
        return self.sps_in_fiber * self.baudrate

    def prepare(self, roll_off, is_cuda=False):
        raise NotImplementedError

    def __getitem__(self, value):
        return self.ds_in_fiber[value]

    def __setitem__(self, key, value):
        self.ds_in_fiber[key] = value

    @property
    def shape(self):
        return self.ds_in_fiber.shape

    def psd(self):
        if self.is_on_cuda:
            self.cpu()
            plt.psd(self.ds_in_fiber[0], NFFT=16384, Fs=self.fs_in_fiber, scale_by_freq=True)
            self.cuda()
            plt.show()
        else:
            plt.psd(self.ds_in_fiber[0], NFFT=16384, Fs=self.fs_in_fiber, scale_by_freq=True)
            plt.show()

    def __len__(self):
        return len(self[0])

    @property
    def constl(self):
        return self.__constl

    @constl.setter
    def constl(self, value):
        self.__constl = value

    def scatterplot(self, sps):
        if self.is_on_cuda:
            self.cpu()
            fignumber = self.shape[0]
            fig, axes = plt.subplots(nrows=1, ncols=fignumber)
            for ith, ax in enumerate(axes):
                ax.scatter(self.ds_in_fiber[ith, ::sps].real, self.ds_in_fiber[ith, ::sps].imag, s=1, c='b')
                ax.set_aspect('equal', 'box')

                ax.set_xlim(
                    [self.ds_in_fiber[ith, ::sps].real.min() - 0.1, self.ds_in_fiber[ith, ::sps].real.max() + 0.1])
                ax.set_ylim(
                    [self.ds_in_fiber[ith, ::sps].imag.min() - 0.1, self.ds_in_fiber[ith, ::sps].imag.max() + 0.1])

            plt.tight_layout()
            plt.show()
            self.cuda()

    @property
    def samples(self):
        return self.ds_in_fiber

    @samples.setter
    def samples(self, value):
        self.ds_in_fiber = value

    @property
    def fs(self):
        return self.baudrate * self.sps

    def cuda(self):
        if self.is_on_cuda:
            return

        try:
            import cupy as cp
        except ImportError:
            return
        self.ds_in_fiber = cp.array(self.ds_in_fiber)
        self.ds = cp.array(self.ds)
        self.is_on_cuda = True

        return self

    def cpu(self):
        if not self.is_on_cuda:
            return

        else:
            import cupy as cp
            self.ds_in_fiber = cp.asnumpy(self.ds_in_fiber)
            self.ds = cp.asnumpy(self.ds_in_fiber)
            self.is_on_cuda = False
        return self

    @property
    def wavelength(self):
        from scipy.constants import c
        return c/self.freq
class QamSignal(Signal):
    def __init__(self, qamorder, baudrate, sps, sps_in_fiber, symbol_length, pol_number):
        '''

        :param qamorder:
        :param baudrate: hz
        :param sps:
        :param sps_in_fiber:
        :param symbol_length:
        :param pol_number:
        '''
        super().__init__(qamorder, baudrate, sps, sps_in_fiber, symbol_length, pol_number)
        self.message = np.random.randint(low=0, high=self.qam_order, size=(self.pol_number, self.symbol_length))
        self.map()

    def map(self):
        import joblib
        constl = joblib.load('constl')[self.qam_order][0]
        self.symbol = np.zeros_like(self.message, dtype=np.complex)
        for row_index, sym in enumerate(self.symbol):
            for i in range(self.qam_order):
                sym[self.message[row_index] == i] = constl[i]

        self.constl = constl

    def prepare(self, roll_off, is_cuda=False):

        self.ds = upsampling(self.symbol, self.sps)
        self.ds_in_fiber = np.zeros((self.pol_number, self.symbol.shape[1] * self.sps_in_fiber),
                                    dtype=self.symbol.dtype)
        if is_cuda:
            self.cuda()

        for index, row in enumerate(self.ds):
            row[:] = rrcos_pulseshaping_freq(row, self.fs, 1 / self.baudrate, roll_off, self.is_on_cuda)
            if not self.is_on_cuda:
                import resampy
                self.ds_in_fiber[index] = resampy.resample(row, self.sps, self.sps_in_fiber, filter='kaiser_fast')
            else:
                import cusignal
                self.ds_in_fiber[index] = cusignal.resample_poly(row, self.sps_in_fiber / self.sps, 1, axis=-1)

    @property
    def time_vector(self):
        return 1 / self.fs_in_fiber * np.arange(self.ds_in_fiber.shape[1])


class WdmSignal(object):
    def __init__(self, symbols, wdm_samples, freq, is_on_cuda, fs_in_fiber):
        self.symbols = symbols
        self.wdm_samples = wdm_samples

        self.relative_freq = freq
        self.is_on_cuda = is_on_cuda
        self.fs_in_fiber = fs_in_fiber

        self.wdm_comb_config = None
        self.baudrates = None
        self.qam_orders = None

        if self.is_on_cuda:
            import cupy as cp
            self.fs_in_fiber = cp.asnumpy(self.fs_in_fiber)
            self.wdm_comb_config = cp.asnumpy(self.wdm_comb_config)

            self.relative_freq = cp.asnumpy(self.relative_freq)

    def cuda(self):
        if self.is_on_cuda:
            return self

        else:
            try:
                import cupy as cp
            except ImportError:
                return self

            self.wdm_samples = cp.array(self.wdm_samples)
            self.is_on_cuda = True

        return self

    def cpu(self):
        if not self.is_on_cuda:
            return self
        else:
            import cupy as cp
            self.wdm_samples = cp.asnumpy(self.wdm_samples)
            self.fs_in_fiber = cp.asnumpy(self.fs_in_fiber)
            self.is_on_cuda = False
        return self

    def __getitem__(self, value):
        return self.wdm_samples[value]

    def __setitem__(self, key, value):
        self.wdm_samples[key] = value

    def psd(self):
        if self.is_on_cuda:
            self.cpu()

            plt.psd(self[0], NFFT=16384, Fs=self.fs_in_fiber, window=np.hamming(16384))
            plt.show()
            self.cuda()
        else:
            plt.psd(self[0], NFFT=16384, Fs=self.fs_in_fiber, window=np.hamming(16384))
            plt.show()
    @property
    def shape(self):
        return self.wdm_samples.shape

    @property
    def __len__(self):
        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np
        return len(np.atleast_2d(self.wdm_samples)[0])


class DummySignal:
    def __init__(self, samples, baudrate, qam_order, symbol, is_on_cuda, fs_in_fiber):
        self.samples = samples
        self.baudrate = baudrate
        self.qam_order = qam_order
        self.symbol = symbol
        self.is_on_cuda = is_on_cuda

        self.fs_in_fiber = fs_in_fiber

        self.sps = None
        
        if self.is_on_cuda:
            import cupy as cp
            self.fs_in_fiber = cp.asnumpy(self.fs_in_fiber)

    @property
    def fs(self):
        assert self.sps is not None
        return self.sps * self.baudrate

    def cpu(self):
        if not self.is_on_cuda:
            return
        else:
            import cupy as cp
            self.samples = cp.asnumpy(self.samples)
            self.is_on_cuda = False

    def cuda(self):
        if self.is_on_cuda:
            return
        else:
            try:
                import cupy as cp
            except ImportError:
                return
            self.samples = cp.array(self.samples)
            self.is_on_cuda = True

    def __getitem__(self, key):
        return self.samples[key]

    def psd(self):
        if self.is_on_cuda:
            self.cpu()
            plt.psd(self[0], NFFT=16384, window=np.hamming(16384))
            plt.show()
            self.cuda()
        else:
            plt.psd(self[0], NFFT=16384, window=np.hamming(16384))
            plt.show()

    @property
    def shape(self):
        return self.samples.shape

    def scatterplot(self, sps):
        if self.is_on_cuda:
            self.cpu()
            fignumber = self.shape[0]
            fig, axes = plt.subplots(nrows=1, ncols=fignumber)
            for ith, ax in enumerate(axes):
                ax.scatter(self[ith, ::sps].real, self[ith, ::sps].imag, s=1, c='b')
                ax.set_aspect('equal', 'box')

                ax.set_xlim([self[ith, ::sps].real.min() - 0.1, self[ith, ::sps].real.max() + 0.1])
                ax.set_ylim([self[ith, ::sps].imag.min() - 0.1, self[ith, ::sps].imag.max() + 0.1])

            plt.tight_layout()
            plt.show()
            self.cuda()
