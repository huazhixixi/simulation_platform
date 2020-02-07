import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from library.channel import NonlinearFiber
from library.receiver_dsp import cd_compensation, matched_filter
from library.receiver_dsp import Superscalar
from library.signal_define import WdmSignal
from library.optics import Demux



span_dict = {
    0:dict(alpha=0.2,gamma=1.3,D=16.7),
    1:dict(alpha=0.17,D=20.1,gamma=0.8),
    2:dict(alpha=0.22,D=3.8,gamma=1.5)
}

config_csv = pd.read_csv('span_power_config.csv')



def generate_span_for_cdc(ith):
    spans = []
    span_config = config_csv.loc[ith]['0th_span':'14th_span']
    for fiber_type in span_config:
        spans.append(NonlinearFiber(**span_dict[fiber_type],length=80,reference_wavelength=1550,slope=0,accuracy='single'))
    return spans



class CoherentReceiver:

    def __init__(self,span,load_ase,ase_snr=None):
        import resampy
        self.load_ase = load_ase
        self.ase_snr = ase_snr
        self.span = span
        self.resampy = resampy
        if self.load_ase:
            assert self.ase_snr is not None

        self.snr_db_res = None
    def prop(self,signal):
        # cd
        signal = cd_compensation(self.span,signal,fs=signal.fs_in_fiber)
        #demux
        signal = Demux.demux_signal(signal,1)

        # resample
        signal.samples = self.resampy.resample(signal[:],signal.sps,2,axis=-1,filter='kaiser_best')
        signal.sps = 2
        signal = matched_filter(signal,0.1)


        #cpe
        signal.samples = signal.samples[:,::2]
        signal.inplace_normalise()
        cpe_alg = Superscalar(200,0.2,20,0,4)
        signal = cpe_alg.prop(signal)
        signal.inplace_normalise()
        # calc Snr
        noise = signal[:] - cpe_alg.symbol_for_snr
        noise_power = (np.abs(noise)**2).mean(axis=-1).sum()

        self.snr_db_res = 10*np.log10((2-noise_power)/noise_power)
        return signal
def main():
    res = {}
    import os
    names = os.listdir('H:/ai/bermargin/withase')
    names = map(lambda x: 'H:/ai/bermargin/withase/' + x, names)
    names = list(names)
    for name in names:
        try:
            signal = WdmSignal.load(name)
            item = int(name.split('/')[-1].split('_')[1])
        except Exception as e:
            print(e)
            print(name)

        fiber = generate_span_for_cdc(item)
        receiver = CoherentReceiver(fiber, False, None)
        signal = receiver.prop(signal)
        res[name] = receiver.snr_db_res
        print(name,receiver.snr_db_res)

main()


