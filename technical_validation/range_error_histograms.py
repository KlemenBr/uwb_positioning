import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


locations = ['location0', 'location1', 'location2', 'location3']

for location in locations:
    print(location)
    data = {}

    with open('./data_set/' + location + '/' + 'data.json', 'r') as f:
        data = json.load(f)

    # load walking path data
    path = data['path']
    channels = data['channels']
    anchors = data['anchors']


    rng_error_los = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}
    rng_error_nlos = {'ch1': [[],[]], 'ch2': [[],[]], 'ch3': [[],[]], 'ch4': [[],[]], 'ch5': [[],[]], 'ch7': [[],[]],}

    for position in path:
        pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
    
        for channel in channels:
            for anchor in anchors:
                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    range_error = item['range'] - range


                    if 1 == item['nlos']:
                        rng_error_nlos[channel][0].append(range)
                        rng_error_nlos[channel][1].append(range_error)
                    else:
                        rng_error_los[channel][0].append(range)
                        rng_error_los[channel][1].append(range_error)


    if 'location0' == location:
        xmin = -1
        xmax = 3
    elif 'location1' == location:
        xmin = -1
        xmax = 2
    elif 'location2' == location:
        xmin = -1
        xmax = 5
    elif 'location3' == location:
        xmin = -1
        xmax = 3
    nbins = 200
    # histograms for LoS
    plt.figure(figsize=(10,6), layout='tight')
    ax = plt.subplot(2,1,1)
    plt.title('LoS Ranging Error [m]')
    plt.hist(rng_error_los['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
    plt.hist(rng_error_los['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
    plt.hist(rng_error_los['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
    plt.hist(rng_error_los['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
    plt.hist(rng_error_los['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
    plt.hist(rng_error_los['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
    plt.xlabel('Ranging error [m]')
    plt.ylabel('Probability')
    plt.grid()
    plt.legend()

    # histograms for NLoS
    ax = plt.subplot(2,1,2)
    plt.title('NLoS Ranging Error [m]')
    plt.hist(rng_error_nlos['ch1'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch1')
    plt.hist(rng_error_nlos['ch2'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch2')
    plt.hist(rng_error_nlos['ch3'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch3')
    plt.hist(rng_error_nlos['ch4'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch4')
    plt.hist(rng_error_nlos['ch5'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch5')
    plt.hist(rng_error_nlos['ch7'][1], range=[xmin,xmax], bins=nbins, density=True, cumulative=False, histtype='stepfilled', label='ch7')
    plt.xlabel('Ranging error [m]')
    plt.ylabel('Probability')
    plt.grid()
    plt.legend()

    filename = './data_set/technical_validation/range_error_histograms/' + location + '.png'
    print('Saving ' + filename)
    plt.savefig(filename)


