import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data2 = {}
data3 = {}

with open('../data_set/environment2/' + 'data.json', 'r') as f:
    data2 = json.load(f)
with open('../data_set/environment3/' + 'data.json', 'r') as f:
    data3 = json.load(f)

# load walking path data
path2 = data2['path']
channels2 = data2['channels']
anchors2 = data2['anchors']
# load walking path data
path3 = data3['path']
channels3 = data3['channels']
anchors3 = data3['anchors']


rng_error_los2 = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}
rng_error_nlos2 = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}
rng_error_los3 = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}
rng_error_nlos3 = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}

for position in path2:
    pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
    
    for channel in channels2:
        for anchor in anchors2:
            for item in data2['measurements'][pos_name][anchor][channel]:
                # calculate euclidean distance
                range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                np.power((item['y_anchor'] - item['y_tag']),2) + 
                                np.power((item['z_anchor'] - item['z_tag']),2))
                range_error = item['range'] - range


                if 1 == item['nlos']:
                    rng_error_nlos2[channel][0].append(range)
                    rng_error_nlos2[channel][1].append(range_error)
                else:
                    rng_error_los2[channel][0].append(range)
                    rng_error_los2[channel][1].append(range_error)

for position in path3:
    pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
    
    for channel in channels3:
        for anchor in anchors3:
            for item in data3['measurements'][pos_name][anchor][channel]:
                # calculate euclidean distance
                range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                np.power((item['y_anchor'] - item['y_tag']),2) + 
                                np.power((item['z_anchor'] - item['z_tag']),2))
                range_error = item['range'] - range


                if 1 == item['nlos']:
                    rng_error_nlos3[channel][0].append(range)
                    rng_error_nlos3[channel][1].append(range_error)
                else:
                    rng_error_los3[channel][0].append(range)
                    rng_error_los3[channel][1].append(range_error)


nbins = 200

# histograms
plt.figure(figsize=(10,6), layout='tight', dpi=300)

xmin = -1
xmax = 5

ax = plt.subplot(2,2,1)
plt.title('a) LoS Ranging Error [m]: environment2')
plt.hist(rng_error_los2['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
plt.hist(rng_error_los2['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
plt.hist(rng_error_los2['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
plt.hist(rng_error_los2['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
plt.hist(rng_error_los2['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
plt.hist(rng_error_los2['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
plt.xlabel('Ranging error [m]')
plt.ylabel('Probability')
plt.grid()
plt.legend()

# histograms for NLoS
ax = plt.subplot(2,2,3)
plt.title('b) NLoS Ranging Error [m]: environment2')
plt.hist(rng_error_nlos2['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
plt.hist(rng_error_nlos2['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
plt.hist(rng_error_nlos2['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
plt.hist(rng_error_nlos2['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
plt.hist(rng_error_nlos2['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
plt.hist(rng_error_nlos2['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
plt.xlabel('Ranging error [m]')
plt.ylabel('Probability')
plt.grid()
plt.legend()


xmin = -1
xmax = 5

ax = plt.subplot(2,2,2)
plt.title('c) LoS Ranging Error [m]: environment3')
plt.hist(rng_error_los3['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
plt.hist(rng_error_los3['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
plt.hist(rng_error_los3['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
plt.hist(rng_error_los3['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
plt.hist(rng_error_los3['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
plt.hist(rng_error_los3['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
plt.xlabel('Ranging error [m]')
plt.ylabel('Probability')
plt.grid()
plt.legend()

# histograms for NLoS
ax = plt.subplot(2,2,4)
plt.title('d) NLoS Ranging Error [m]: environment3')
plt.hist(rng_error_nlos3['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
plt.hist(rng_error_nlos3['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
plt.hist(rng_error_nlos3['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
plt.hist(rng_error_nlos3['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
plt.hist(rng_error_nlos3['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
plt.hist(rng_error_nlos3['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
plt.xlabel('Ranging error [m]')
plt.ylabel('Probability')
plt.grid()
plt.legend()


filename = '../data_set/technical_validation/range_error_histograms/loc2_loc3.png'
print('Saving ' + filename)
plt.savefig(filename)


