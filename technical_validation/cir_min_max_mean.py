import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

CIRLEN = 152
OFFSET = 5

#channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
#anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
environments = ['environment0', 'environment1', 'environment2', 'environment3']




def cpx2abs(complexlist):
	""" convert numpy array of complex number strings to a numpy array of absolute values"""
	temp = np.empty((len(complexlist)), dtype=complex)
	for i in range(temp.shape[0]):
		temp[i] = complex(complexlist[i])
	abs_data = np.absolute(temp)

	return abs_data


def generate_cir(data):
    """
    Find useful CIR index limits inside the input data
    """
    rxpacc = data['rxpacc']
    fp_index = int(data['fp_index'])
    startidx = fp_index - OFFSET
    abs_data = (cpx2abs(data['cir'])/float(rxpacc))[startidx:startidx+CIRLEN]

    return abs_data



for environment in environments:
    print(environment)
    data = {}

    with open('../data_set/' + environment + '/' + 'data.json', 'r') as f:
        data = json.load(f)
    
    # create directory if it doesnt exist
    os.makedirs('../data_set/technical_validation/cir_min_max_mean/' + environment, exist_ok=True)

    # load walking path data
    path = data['path']
    channels = data['channels']
    anchors = data['anchors']
    

    for channel in channels:

        for anchor in anchors:
            # get all CIRs
            cir = []
            for position in path:
                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

                for item in data['measurements'][pos_name][anchor][channel]:
                    cirtemp = generate_cir(item)
                    cir.append(cirtemp)
                    
            # max min mean
            cir = np.array(cir)

            cirmax = cir.max(axis=0)
            cirmin = cir.min(axis=0)
            cirmean = cir.mean(axis=0)
            
            # calculate symbol time
            prfr = 64 # 64MHz
            if prfr == 16:
                symbol_time = 496/499.2 * 1e-6 #us
                acc_sample_time = symbol_time / 992
            else:
                symbol_time = 508/499.2 * 1e-6 #us
                acc_sample_time = symbol_time / 1016
            
            # define time axis for CIR and PDP
            ts = np.arange(-OFFSET*acc_sample_time, (CIRLEN-OFFSET)*acc_sample_time, acc_sample_time)*1e9

            plt.figure(figsize=(10,6), layout='tight')
            plt.title('CIR ' + anchor + ' ' + channel)
            ax = plt.subplot(1,1,1)
            plt.plot(ts, cirmean, color='white', label='mean(CIR)')
            plt.fill_between(x=ts, y1=cirmin, y2=cirmax, label='min-max(CIR)')
            plt.xlabel('Time [ns]')
            plt.ylabel('Absolute value')
            plt.grid()
            plt.legend()


            filename = '../data_set/technical_validation/cir_min_max_mean/' + environment + '/' + anchor + '_' + channel + '.png'
            print('Saving ' + filename)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()



