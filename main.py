# import numpy as np
# import matplotlib.pyplot as plt
# from library.optics import Laser
# from library.channel import NonlinearFiber
# from library.signal_define import QamSignal
# import tqdm
# np.random.seed(0)
# # from BER get OSNR margin
# # 只有非线性
# def only_nonlinear_prop(savedir,baudrate,,laser_power_dbm,link):
#
#     signal = QamSignal(16,baudrate,sps=2,sps_in_fiber=4,symbol_length = 65536, pol_number = 2)
#     signal.prepare(0.2,False)
#     laser = Laser(0,False,193.1e12,laser_power_dbm)
#
#     for qijian in link:
#         signal = qijian.prop(signal)
#         signal.save(savedir)
#
#
# def main():
#     import pandas as pd
#     config = pd.read('config.csv')
#
#     baudrate = config.baudrate
#     laser_power_dbm = config.laser_power_dbm
#     span_number = config.span_number
#     links = generate(span_number)
#
# def generate_config():
#     import random
#     # 随机选取光纤类型，0，1，2 代表三种光纤类型
#     # 光纤链路固定为15
#     # 光纤span长度固定为80km
#
#     type = np.random.randint(0,3,size=(1,15))
#     span_type = ['SSMF','TWC','ELEAF']
#
#
#
#
#
#
#
#
# %%

from library.signal_define import QamSignal, WdmSignal
from library.optics import Mux, Laser, Edfa
from library.channel import NonlinearFiber


def generate_wdm_signal(nch=3):
    lasers = [Laser(0, False, 193.45e12 + i * 50e9, 0) for i in range(3)]
    signals = [QamSignal(16, 35e9, 2, 8, 65536, 2) for _ in range(3)]

    for laser, sig in zip(lasers, signals):
        sig = sig.prepare(0.1, True)
        sig = laser.prop(sig)

    wdm_signal = Mux.mux_signal(signals)
    return wdm_signal




from library.channel import NonlinearFiber
import numpy as np
np.random.seed(0)
import tqdm

wdm_signal = generate_wdm_signal(3)
wdm_signal.to_32complex()
#wdm_signal.save_to_mat('before_transimt')
fiber = NonlinearFiber(0.2,16.7,80,1550,0,'single')

for i in tqdm.tqdm(range(5)):
    wdm_signal = fiber.prop(wdm_signal)
    wdm_signal[:] = np.sqrt(10**(16/10))*wdm_signal[:]
wdm_signal.save_to_mat('after_transimt_single_accuracy')










# %%




