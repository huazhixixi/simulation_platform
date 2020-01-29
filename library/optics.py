import numpy as np
import matplotlib.pyplot as plt
from .signal_define import WdmSignal
from .filter_design import  ideal_lp

class Laser(object):

    def __init__(self, linewidth, is_phase_noise, freq,laser_power):
        '''
            linewidth:hz
            freq:hz
        '''
        self.linewidth = linewidth
        self.is_phase_noise = is_phase_noise
        self.freq = freq
        self.is_on_cuda = False
        self.laser_power = laser_power
        
    def phase_noise(self, signal):

        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        var = 2 * np.pi * self.linewidth / signal.fs_in_fiber
        f = np.random.normal(scale=np.sqrt(var), size=signal[:].shape)
        for row in f:
            row[0] = (np.random.rand(1) * 2 * np.pi - np.pi)[0]
        return np.cumsum(f, axis=1)

    def prop(self, signal):
        signal.freq = self.freq
        self.is_on_cuda = signal.is_on_cuda

        if self.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        if self.is_phase_noise:
            self.phase_noise = self.phase_noise(signal)
            signal[:] = signal[:] * np.exp(1j * self.phase_noise)
            
        signal.inplace_normalise()
        signal.set_signal_power(self.laser_power)
        return signal

    def plot_phase_noise(self):

        if self.is_phase_noise:
            if self.is_on_cuda:
                import cupy as cp
                self.phase_noise = cp.asnumpy(self.phase_noise)

            fig, axes = plt.subplots(figsize=(10, 2), nrows=1, ncols=self.phase_noise.shape[0])
            for ith, ax in enumerate(axes):
                ax.plot(self.phase_noise[ith], c='b', lw=2)

        plt.tight_layout()
        plt.show()


class WSS(object):

    def __init__(self, frequency_offset, bandwidth, oft):

        '''

        :param frequency_offset: value away from center [GHz]
        :param bandwidth: 3-db Bandwidth [Ghz]
        :param oft:GHZ
        '''
        self.frequency_offset = frequency_offset / 1e9
        self.bandwidth = bandwidth / 1e9
        self.oft = oft / 1e9
        self.H = None
        self.freq = None
        self.is_on_cuda = False

    def prop(self, signal):

        sample = signal[:]
        self.is_on_cuda = signal.is_on_cuda

        if self.is_on_cuda:
            import cupy as np

        else:
            import numpy as np

        freq = np.fft.fftfreq(len(sample[0, :]), 1 / signal.fs_in_fiber)
        freq = freq / 1e9

        self.freq = freq
        self.__get_transfer_function(freq)

        for i in range(sample.shape[0]):
            sample[i, :] = np.fft.ifft(np.fft.fft(sample[i, :]) * self.H)

        signal[:] = sample
        return signal

    def __get_transfer_function(self, freq_vector):
        if self.is_on_cuda:
            import cupy as np
            from cupyx.scipy.special import erf
        else:
            import numpy as np
            from scipy.special import erf

        delta = self.oft / 2 / np.sqrt(2 * np.log(2))

        H = 0.5 * delta * np.sqrt(2 * np.pi) * (
                erf((self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta) - erf(
            (-self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta))

        H = H / np.max(H)

        self.H = H

    def plot_transfer_function(self, freq=None):
        import matplotlib.pyplot as plt
        if self.H is None:
            self.__get_transfer_function(freq)
            self.freq = freq

        if self.is_on_cuda:
            import cupy as cp
            self.H = cp.asnumpy(self.H)
            self.freq = cp.asnumpy(self.freq)

        index = self.H > 0.001
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.scatter(self.freq[index], np.abs(self.H[index]), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("without log")
        plt.subplot(122)
        plt.scatter(self.freq[index], 10 * np.log10(np.abs(self.H[index])), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("with log")
        plt.show()
        if self.is_on_cuda:
            self.H = cp.array(self.H)
            self.freq = cp.array(self.freq)

    def __str__(self):

        string = f'the center_frequency is {0 + self.frequency_offset}[GHZ] \t\n' \
                 f'the 3-db bandwidth is {self.bandwidth}[GHz]\t\n' \
                 f'the otf is {self.oft} [GHz] \t\n'
        return string

    def __repr__(self):
        return self.__str__()


class Mux(object):

    @staticmethod
    def mux_signal(signals, relative_freq=None, wdm_comb_config=None):
        if signals[0].is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        freqs = np.array([signal.freq for signal in signals])

        fs = np.array([signal.fs_in_fiber for signal in signals])

        if not np.all(np.diff(fs) == 0):
            print(np.diff(fs))
            raise Exception('fs_in_fiber of the signal must be the same')

        length = np.array([signal.shape[1] for signal in signals])

        if relative_freq is None:
            relative_freq = np.array(freqs) - (np.max(freqs) + np.min(freqs)) / 2
            wdm_comb_config = np.arange(len(signals))
        else:
            assert wdm_comb_config is not None

        max_length = np.max(length)
        df = fs[0] / max_length

        wdm_samples = 0


        for idx, signal in enumerate(signals):
            freq_samples = np.fft.fft(signal[:], n=max_length, axis=-1)
            yidong_dianshu = relative_freq[idx] / df
            yidong_dianshu = np.ceil(yidong_dianshu)
            yidong_dianshu = np.int(yidong_dianshu)
            freq_samples = np.roll(freq_samples, yidong_dianshu, axis=-1)

            wdm_samples += freq_samples
        symbols = [signal.symbol for signal in signals]
        wdm_signal = WdmSignal(symbols, np.fft.ifft(wdm_samples, axis=-1), relative_freq, signals[0].is_on_cuda,
                               fs_in_fiber=fs[0])
        wdm_signal.wdm_comb_config = wdm_comb_config
        wdm_signal.baudrates = [signal.baudrate for signal in signals]
        wdm_signal.qam_orders = [signal.qam_order for signal in signals]
        return wdm_signal


class Demux(object):

    @staticmethod
    def demux_signal(wdm_signals, signal_index):
        if wdm_signals.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        index = wdm_signals.wdm_comb_config.tolist().index(signal_index)
        relative_freq = wdm_signals.relative_freq[index]

        df = wdm_signals.fs_in_fiber / wdm_signals.length

        freq_samples = np.fft.fft(wdm_signals[:], axis=-1)
        yidong_dianshu = relative_freq / df
        yidong_dianshu = np.ceil(yidong_dianshu)
        yidong_dianshu = np.int(yidong_dianshu)
        freq_samples = np.roll(freq_samples, -yidong_dianshu, axis=-1)

        # ideal low pass filter
        left_freq = -wdm_signals.baudrates[index] / 2
        right_freq = -left_freq
        freq_samples = ideal_lp(freq_samples, left_freq, right_freq, wdm_signals.fs_in_fiber, need_fft=False)

        symbols = wdm_signals.symbols[index]
        baudrate = wdm_signals.baudrates[index]
        qam_order = wdm_signals.qam_orders[index]
        signal = DummySignal(np.fft.ifft(freq_samples, axis=-1), baudrate, qam_order, symbols, wdm_signals.is_on_cuda,
                             wdm_signals.fs_in_fiber)
        return signal

