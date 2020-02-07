import numpy as np
import matplotlib.pyplot as plt
from library.optics import Laser, Mux,ConstantGainEdfa
from library.channel import NonlinearFiber
from library.signal_define import QamSignal
import tqdm
import  tqdm
np.random.seed(0)
# from BER get OSNR margin
# 只有非线性


span_dict = {
    0:dict(alpha=0.2,gamma=1.3,D=16.7),
    1:dict(alpha=0.17,D=20.1,gamma=0.8),
    2:dict(alpha=0.22,D=3.8,gamma=1.5)
}


def generate_wdm_signal(powerdbm,nch=3):
    lasers = [Laser(0, False, 193.45e12 + i * 50e9, powerdbm) for i in range(3)]
    signals = [QamSignal(16, 35e9, 2, 8, 65536, 2) for _ in range(3)]

    for laser, sig in zip(lasers, signals):
        sig = sig.prepare(0.1, True)
        sig = laser.prop(sig)

    wdm_signal = Mux.mux_signal(signals)
    return wdm_signal


def only_nonlinear_prop(savedir):
    import pandas as pd
    data = pd.read_csv('span_power_config.csv')
    powers = data.power.values.reshape(-1,1)
    span_configs = data.loc[:,'0th_span':'14th_span'].values

    for itemindex in tqdm.tqdm(range(powers.shape[0]//2)):
        spans = []
        edfas = []
        power = powers[itemindex,0]
        span = span_configs[itemindex]


        wdm_signal = generate_wdm_signal(powerdbm=power)
        wdm_signal.to_32complex()
        for fiber_type in span:
            spans.append(NonlinearFiber(**span_dict[fiber_type],length=80,reference_wavelength=1550,slope=0,accuracy='single'))
            edfas.append(ConstantGainEdfa(gain=spans[-1].alpha*spans[-1].length,nf=5,is_ase=True))

        for span,edfa in zip(spans,edfas):

            wdm_signal = span.prop(wdm_signal)
            wdm_signal = edfa.prop(wdm_signal)
        wdm_signal.save(savedir+f'item_{itemindex}_ase')


def main():
    import os
    only_nonlinear_prop(r'H:/ai/bermargin/withase/')





main()



def generate_config():
    import random
    # 随机选取光纤类型，0，1，2 代表三种光纤类型
    # 光纤链路固定为15
    # 光纤span长度固定为80km

    type = np.random.randint(0,3,size=(1,15))
    span_type = ['SSMF','TWC','ELEAF']



def get_optimum_power_from_gnmodel(span_config_file):
    import pandas as pd
    from library.gn_model import Span as GnSpan
    from library.gn_model import Signal as GnSignal
    from library.gn_model import Edfa as GnEdfa
    span_config_file = pd.read_csv(span_config_file)
    chose_power = []

    item = span_config_file.values
    for index in tqdm.tqdm(range(item.shape[0])):
        spans = []
        edfas = []
        snr = []
        oneitem = item[index]
        for fiber_type in oneitem:
            spans.append(GnSpan(**span_dict[fiber_type]))
            edfas.append(GnEdfa(gain=spans[-1].alpha * spans[-1].length,nf=5))

        signal_power = np.arange(-5,5,0.1)
        for power in signal_power:
            signals = [
                GnSignal(signal=(10**(power/10)/1000), nli=0, ase=0, carri=193.1e12 + j * 50e9, baudrate=35e9, number=j, mf='dp-16qam')

                for j in range(3)]

            center_channel = signals[int(np.floor(len(signals) / 2))]

            for span,edfa in zip(spans,edfas):
                span.prop(center_channel, signals)
                edfa.prop(center_channel)
                # snr.append((center_channel.nli + 0))


            snr.append(center_channel.signal / (center_channel.nli + center_channel.ase))

            if len(snr)>=2:
                if snr[-1]<snr[-2]:
                    break

            # print(center_channel.nli)
        snr = np.array(snr)
        chose_power.append(signal_power[snr.argmax()])

    span_config_file['power'] = np.array(chose_power)
    span_config_file.to_csv('span_power_config.csv',index=None)
















#
# from library.signal_define import QamSignal, WdmSignal
# from library.optics import Mux, Laser, Edfa
# from library.channel import NonlinearFiber
#
#
# def generate_wdm_signal(nch=3):
#     lasers = [Laser(0, False, 193.45e12 + i * 50e9, 0) for i in range(3)]
#     signals = [QamSignal(16, 35e9, 2, 8, 65536, 2) for _ in range(3)]
#
#     for laser, sig in zip(lasers, signals):
#         sig = sig.prepare(0.1, True)
#         sig = laser.prop(sig)
#
#     wdm_signal = Mux.mux_signal(signals)
#     return wdm_signal
#
#
#
#
# from library.channel import NonlinearFiber
# import numpy as np
# np.random.seed(0)
# import tqdm
#
# wdm_signal = generate_wdm_signal(3)
# wdm_signal.to_32complex()
# #wdm_signal.save_to_mat('before_transimt')
# fiber = NonlinearFiber(0.2,16.7,80,1550,0,'single')
#
# for i in tqdm.tqdm(range(5)):
#     wdm_signal = fiber.prop(wdm_signal)
#     wdm_signal[:] = np.sqrt(10**(16/10))*wdm_signal[:]
# wdm_signal.save_to_mat('after_transimt_single_accuracy')
#









# %%




