import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

CIRLEN = 152
OFFSET = 5

# LoS=8
# nLoS=61

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



environment = 'environment2'
data = {}

with open('../data_set/' + environment + '/' + 'data.json', 'r') as f:
    data = json.load(f)
    
# load walking path data
path = data['path']
channels = data['channels']
anchors = data['anchors']
    
channel = 'ch1'
anchor = 'A1'

los = []
position = path[8]
pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

for item in data['measurements'][pos_name][anchor][channel]:
    cirtemp = generate_cir(item)
    los.append(cirtemp)
                    
los = np.array(los)
losmean = los.mean(axis=0)
                
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
plt.title(environment + ' ' + anchor + ' ' + channel + ' ' + 'LoS')
ax = plt.subplot(1,1,1)
plt.plot(ts, losmean, label='CIR')
plt.xlabel('Time [ns]')
plt.ylabel('Absolute value')
plt.grid()
plt.legend()


filename = '../data_set/technical_validation/los_nlos/' + anchor + '_' + channel + '_' + 'los' + '.png'
print('Saving ' + filename)
plt.savefig(filename, bbox_inches='tight')
plt.close()


nlos = []
position = path[61]
pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

for item in data['measurements'][pos_name][anchor][channel]:
    cirtemp = generate_cir(item)
    nlos.append(cirtemp)
                    
nlos = np.array(nlos)
nlosmean = nlos.mean(axis=0)
                
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
plt.title(environment + ' ' + anchor + ' ' + channel + ' ' + 'nLoS')
ax = plt.subplot(1,1,1)
plt.plot(ts, nlosmean, label='CIR')
plt.xlabel('Time [ns]')
plt.ylabel('Absolute value')
plt.grid()
plt.legend()


filename = '../data_set/technical_validation/los_nlos/' + anchor + '_' + channel + '_' + 'nlos' + '.png'
print('Saving ' + filename)
plt.savefig(filename, bbox_inches='tight')
plt.close()





plt.figure(figsize=(14,6), dpi=300, layout='tight')
plt.title(environment + ' ' + anchor + ' ' + channel + ' ' + 'nLoS')
ax = plt.subplot(1,2,1)
plt.plot(ts, losmean, label='CIR')
plt.title('a) LoS CIR')
plt.xlabel('Time [ns]')
plt.ylabel('Absolute value')
plt.grid()
plt.legend()
ax = plt.subplot(1,2,2)
plt.plot(ts, nlosmean, label='CIR')
plt.title('b) nLoS CIR')
plt.xlabel('Time [ns]')
plt.ylabel('Absolute value')
plt.grid()
plt.legend()


filename = '../data_set/technical_validation/los_nlos/los_nlos.png'
print('Saving ' + filename)
plt.savefig(filename, bbox_inches='tight')
plt.close()