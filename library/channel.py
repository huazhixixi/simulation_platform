import numpy as np
from scipy.fft import fft, ifft, fftfreq

from scipy.constants import c


class AwgnChannel(object):

    def __init__(self, snr):
        self.snr = snr
        self.snr_linear = 10 ** (self.snr / 10)

    def prop(self, signal, power='measure'):
        if signal.is_on_cuda:
            import cupy as np
        else:
            import numpy as np

        if power.lower() == 'measure':
            
            power = np.mean(np.abs(signal[:]) ** 2, axis=1)
            power = np.sum(power)
            noise_power = power / self.snr_linear * signal.sps_in_fiber
            noise_power_xpol = noise_power / 2
            seq = (noise_power_xpol / 2) * (np.random.randn(2, len(signal)) + 1j * np.random.randn(2, len(signal)))

            signal[:] = signal[:] + seq
            return signal


class Fiber(object):

    def __init__(self, alpha, beta, length, ref):
        self.alpha = alpha
        self.beta = beta
        self.length = length
        self.reference_wavelength = ref  # nm

    def prop(self, signal):
        raise NotImplementedError

    @property
    def alphalin(self):
        alphalin = self.alpha / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''

        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wavelength * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength * 1e-12 * self.D + (
                self.reference_wavelength * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        '''

        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length


class NonlinearFiber(Fiber):

    def __init__(self, **kwargs):
        super(NonlinearFiber, self).__init__(**kwargs)
        self.step_length = kwargs.get('step_length', 20 / 1000)
        self.gamma = kwargs.get('gamma', 1.3)

    @property
    def step_length_eff(self):
        return (1 - self.np.exp(-self.alphalin * self.step_length)) / self.alphalin

    def prop(self, signal):
        signal.cuda()
        if signal.is_on_cuda:
            self.fft = cupyx.scipy.fft
            self.ifft = cupyx.scipy.ifft
            self.plan = cupyx.scipy.plan

        nstep = self.length / self.step_length
        nstep = int(np.floor(nstep))
        freq = fftfreq(signal.shape[1], 1 / signal.fs_in_fiber)
        omeg = 2 * self.np.pi * freq
        D = -1j / 2 * self.beta2(signal.center_wavelength) * omeg ** 2
        N = 8 / 9 * 1j * self.gamma
        atten = -self.alphalin / 2
        last_step = self.length - self.step_length * nstep

        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)
        signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
        signal[0] = signal[0] * self.np.exp(atten * self.step_length / 2)
        signal[1] = signal[1] * self.np.exp(atten * self.step_length / 2)

        for _ in range(nstep - 1):
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length)

            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
            signal[0] = signal[0] * self.np.exp(atten * self.step_length)
            signal[1] = signal[1] * self.np.exp(atten * self.step_length)

        signal[0] = signal[0] * self.np.exp(atten * self.step_length / 2)
        signal[1] = signal[1] * self.np.exp(atten * self.step_length / 2)
        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)

        if last_step:
            last_step_eff = (1 - self.np.exp(-self.alphalin * last_step)) / self.alphalin
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)
            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1], last_step_eff)
            signal[0] = signal[0] * self.np.exp(atten * last_step)
            signal[1] = signal[1] * self.np.exp(atten * last_step)
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)

    def nonlinear_prop(self, N, time_x, time_y, step_length=None):
        if step_length is None:
            time_x = time_x * self.np.exp(
                N * self.step_length_eff * (self.np.abs(time_x) ** 2 + self.np.abs(
                    time_y) ** 2))
            time_y = time_y * self.np.exp(
                N * self.step_length_eff * (self.np.abs(time_x) ** 2 + self.np.abs(time_y) ** 2))
        else:
            time_x = time_x * self.np.exp(
                N * step_length * (self.np.abs(time_x) ** 2 + self.np.abs(
                    time_y) ** 2))
            time_y = time_y * self.np.exp(
                N * step_length * (self.np.abs(time_x) ** 2 + self.np.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop(self, D, timex, timey, length):
        with self.plan:
            freq_x = self.fft(timex, overwrite_x=True)
            freq_y = self.fft(timey, overwrite_x=True)

            freq_x = freq_x * self.np.exp(D * length)
            freq_y = freq_y * self.np.exp(D * length)

            time_x = self.ifft(freq_x, overwrite_x=True)
            time_y = self.ifft(freq_y, overwrite_x=True)
            return time_x, time_y
