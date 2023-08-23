import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import interpolate

def calculate_rss_spline():
    """
    This function generates spline representation of actual RSS vs. estimated RSS
    :return: interpolate.spline object
    """
    """x = [-150, -149, -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133,
     -132, -131, -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
     -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102.3, -102, -100.5, -99, -98, -97,
     -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86.5, -85.7, -85, -84.5, -83.5, -83, -82.5, -82, -81.5,
     -81, -80.8, -80.5, -80.2, -80.1, -79.9, -79.8, -79.6, -79.5, -79.3, -79.2, -78.9]
    y = range(-150, -65, 1)
    """
    x = [-150, -149, -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133,
     -132, -131, -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
     -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102.3, -102, -100.5, -99, -98, -97,
     -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86.5, -85.7, -85, -84.5, -83.5, -83, -82.5, -82, -81.5,
     -81, -80.8, -80.5, -80.2, -80.1, -79.9, -79.8, -79.6, -79.5, -79.3, -79.2, -78.9, -78.73, -78.56, -78.39,
      -78.21, -78.04, -77.87, -77.70, -77.53, -77.36, -77.19, -77.01, -76.84, -76.67, -76.50, -76.33, -76.16, -75.98]
    y = range(-150, -48, 1)

    spl = interpolate.interp1d(x, y, kind='cubic', fill_value=(-150, -48), bounds_error=False)

    return spl


# Recursive defaultdict
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

# create weighting according to the position std instead of 10 bins weighting

def path_loss_exponent(data):
    # number of bins
    N = 10
    hist, edges = np.histogram(data[:,0], bins=N)
    # construct sample bins
    b = np.empty(N, dtype=object)
    for i in range(b.shape[0]):
        b[i] = []
    for i in range(len(data)):
        if (data[i,0] >= edges[0]) & (data[i,0] < edges[1]):
            b[0].append(data[i,1])
        elif (data[i,0] >= edges[1]) & (data[i,0] < edges[2]):
            b[1].append(data[i,1])
        elif (data[i,0] >= edges[2]) & (data[i,0] < edges[3]):
            b[2].append(data[i,1])
        elif (data[i,0] >= edges[3]) & (data[i,0] < edges[4]):
            b[3].append(data[i,1])
        elif (data[i,0] >= edges[4]) & (data[i,0] < edges[5]):
            b[4].append(data[i,1])
        elif (data[i,0] >= edges[5]) & (data[i,0] < edges[6]):
            b[5].append(data[i,1])
        elif (data[i,0] >= edges[6]) & (data[i,0] < edges[7]):
            b[6].append(data[i,1])
        elif (data[i,0] >= edges[7]) & (data[i,0] < edges[8]):
            b[7].append(data[i,1])
        elif (data[i,0] >= edges[8]) & (data[i,0] < edges[9]):
            b[8].append(data[i,1])
        elif (data[i,0] >= edges[9]) & (data[i,0] <= edges[10]):
            b[9].append(data[i,1])
    for l in b:
        l = np.asarray(l)
    b = np.asarray(b)
    
    # weighting array
    w = []
    # calculate weights --> standard deviations of values in bins
    for i in range(b.shape[0]):
        tmp = 1/np.std(b[i])
        # prevent bins with 0 samples to impact the outcomes
        if tmp > 100.0:
            w.append(0.01)
        else:
            w.append(tmp)
    w = np.asarray(w)

    # construct weighting array
    warr = []
    for mer in data[:,0]:
        if (mer >= edges[0]) & (mer < edges[1]):
            warr.append(w[0])
        elif (mer >= edges[1]) & (mer < edges[2]):
            warr.append(w[1])
        elif (mer >= edges[2]) & (mer < edges[3]):
            warr.append(w[2])
        elif (mer >= edges[3]) & (mer < edges[4]):
            warr.append(w[3])
        elif (mer >= edges[4]) & (mer < edges[5]):
            warr.append(w[4])
        elif (mer >= edges[5]) & (mer < edges[6]):
            warr.append(w[5])
        elif (mer >= edges[6]) & (mer < edges[7]):
            warr.append(w[6])
        elif (mer >= edges[7]) & (mer < edges[8]):
            warr.append(w[7])
        elif (mer >= edges[8]) & (mer < edges[9]):
            warr.append(w[8])
        elif (mer >= edges[9]) & (mer <= edges[10]):
            warr.append(w[9])
    warr = np.asarray(warr)
    
    # linear fit [range, rss]
    coef, residuals, rank, singular_values, rcond = np.polyfit(data[:,0], data[:,1], deg=1, w=warr, full=True)

    plt.scatter(data[:,0], data[:,1])
    plt.plot(data[:,0], (data[:,0]*coef[0] + coef[1]), '-k')
    plt.show()

    return coef





environments = ['environment0', 'environment1', 'environment2', 'environment3']


#path_loss_table = {}
path_loss_table = recursive_defaultdict()

spl = calculate_rss_spline()



for environment in environments:
    print(environment)
    data = {}

    with open('../data_set/' + location + '/' + 'data.json', 'r') as f:
        data = json.load(f)

    # load walking path data
    path = data['path']
    anchors = data['anchors']
    channels = data['channels']
    print(len(path))

    # for each channel:
    #   + get path loss for all measurements for the channel
    #   + define weights on individual anchor+position measurements
    #   + add weights to the los/nlos arrays in a form [rng, rss, w]
    for channel in channels:
        los = []
        nlos = []
        
        for anchor in anchors:
            
            for position in path:
                pos = []
                
                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
                NLOS = None

                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    rng = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    # set nlos
                    NLOS = item['nlos']
                    
                    # interpolate RSS
                    rss = spl(item['rss'])
                    pos.append([rng, rss])

                
                # calculate weights and append items to los/nlos arrays
                w = 1/np.std(pos)
                if 0 == NLOS:
                    for item in pos:
                        los.append([rng, rss, w])
                elif 1 == NLOS:
                    for item in pos:
                        nlos.append([rng, rss, w])
        
        los = np.array(los)
        nlos = np.array(nlos)
            
        # calculate path loss exponent
        #ploss_los = path_loss_exponent(los)
        #ploss_nlos = path_loss_exponent(nlos)

        print(channel, anchor)

        # linear fit [range, rss]
        coef_los, residuals, rank, singular_values, rcond = np.polyfit(los[:,0], los[:,1], deg=1, w=los[:,2], full=True)
        plt.scatter(los[:,0], los[:,1])
        plt.plot(los[:,0], (los[:,0]*coef_los[0] + coef_los[1]), '-k')
        plt.show()

        coef_nlos, residuals, rank, singular_values, rcond = np.polyfit(nlos[:,0], nlos[:,1], deg=1, w=nlos[:,2], full=True)
        plt.scatter(nlos[:,0], nlos[:,1])
        plt.plot(nlos[:,0], (nlos[:,0]*coef_nlos[0] + coef_nlos[1]), '-k')
        plt.show()

        print(coef_los)
        print(coef_nlos)
        print()

        #path_loss_table[environment][channel][anchor]['los'] = {'n': ploss_los[0], 'c': ploss_los[1]}
        #path_loss_table[environment][channel][anchor]['nlos'] = {'n': ploss_nlos[0], 'c': ploss_nlos[1]}


        #path_loss_table[environment][channel]['los'] = {'n': ploss_los[0], 'c': ploss_los[1]}
        #path_loss_table[environment][channel]['nlos'] = {'n': ploss_nlos[0], 'c': ploss_nlos[1]}
            

print(path_loss_table)               


   
