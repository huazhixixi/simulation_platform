from .filter_design import rrcos_pulseshaping_freq
from .numba_core import cma_equalize_core
import numpy as np
import numba  

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



class Equalizer(object):
    def __init__(self,ntaps,lr,loops):
        self.wxx = np.zeros((1,ntaps),dtype = np.complex)
        self.wxy = np.zeros((1,ntaps),dtype = np.complex)

        self.wyx = np.zeros((1,ntaps),dtype = np.complex)

        self.wyy = np.zeros((1,ntaps),dtype = np.complex)

        self.wxx[0,ntaps//2] = 1
        self.wyy[0,ntaps//2] = 1
        
        self.ntaps = ntaps
        self.lr = lr
        self.loops = loops
        self.error_xpol_array = None
        self.error_ypol_array = None
        
        self.equalized_symbols = None
        
    def equalize(self,signal):
        
        raise NotImplementedError
        
    def scatterplot(self,sps=1):
        import matplotlib.pyplot as plt
        fignumber = self.equalized_symbols.shape[0]
        fig,axes = plt.subplots(nrows = 1,ncols = fignumber)
        for ith,ax in enumerate(axes):
            ax.scatter(self.equalized_symbols[ith,::sps].real,self.equalized_symbols[ith,::sps].imag,s=1,c='b')
            ax.set_aspect('equal', 'box')

            ax.set_xlim([self.equalized_symbols[ith,::sps].real.min()-0.1,self.equalized_symbols[ith,::sps].real.max()+0.1])
            ax.set_ylim([self.equalized_symbols[ith,::sps].imag.min()-0.1,self.equalized_symbols[ith,::sps].imag.max()+0.1])

        plt.tight_layout()
        plt.show()
    
    def plot_error(self):
        fignumber = self.equalized_symbols.shape[0]
        fig,axes = plt.subplots(figsize=(8,4),nrows = 1,ncols = fignumber)
        for ith,ax in enumerate(axes):
            ax.plot(self.error_xpol_array[0],c='b',lw=1)
        plt.tight_layout()
        plt.show()
        
    def plot_freq_response(self):
        from scipy.fftpack import fft,fftshift
        freq_res =  fftshift(fft(self.wxx)),fftshift(fft(self.wxy)),fftshift(fft(self.wyx)),fftshift(fft(self.wyy))
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,2)
        for idx,row in enumerate(axes.flatten()):
            row.plot(np.abs(freq_res[idx][0]))
            row.set_title(f"{['wxx','wxy','wyx','wyy'][idx]}")
        plt.tight_layout()
        plt.show()
        
    def freq_response(self):
        from scipy.fftpack import fft,fftshift
        freq_res =  fftshift(fft(self.wxx)),fftshift(fft(self.wxy)),fftshift(fft(self.wyx)),fftshift(fft(self.wyy))
        return freq_res

    
class CMA(Equalizer):
    
    
    def __init__(self,ntaps,lr,loops=3):
        super().__init__(ntaps,lr,loops)
       
        
    
    def equalize(self,signal):
        signal.cpu()
        import numpy as np
            
        samples_xpol = _segment_axis(signal[0],self.ntaps, self.ntaps-signal.sps)
        samples_ypol = _segment_axis(signal[1],self.ntaps, self.ntaps-signal.sps)
        
        self.error_xpol_array = np.zeros((self.loops,len(samples_xpol)))
        self.error_ypol_array = np.zeros((self.loops,len(samples_xpol)))
        
        for idx in range(self.loops):
            symbols, self.wxx, self.wxy, self.wyx, \
            self.wyy, error_xpol_array, error_ypol_array \
            = cma_equalize_core(samples_xpol,samples_ypol,\
                                self.wxx,self.wyy,self.wxy,self.wyx,self.lr)
            
            self.error_xpol_array[idx] = np.abs(error_xpol_array[0])**2
            self.error_ypol_array[idx] = np.abs(error_ypol_array[0])**2
        
        self.equalized_symbols = symbols

    
   