import numpy as np 
import matplotlib.pyplot as plt
from library.optics import Laser
from library.channel import NonlinearFiber
from library.signal_define import QamSignal

np.random.seed(0)
# from BER get OSNR margin
# 只有非线性
def only_nonlinear_prop(savedir,baudrate,signalpower,laser_power_dbm,link):
    
    signal = QamSignal(16,baudrate,sps=2,sps_in_fiber=4,symbol_length = 65536, pol_number = 2)
    signal.prepare(0.2,False)
    laser = Laser(0,False,193.1e12,laser_power_dbm)

    for qijian in link:
        signal = qijian.prop(signal)
        
    


