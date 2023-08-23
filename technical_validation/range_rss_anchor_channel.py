import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import interpolate
import math
from sklearn.cluster import DBSCAN

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


def estimate_range(rss, rss_d0, n):
    print(rss)
    print(rss_d0)
    print(n)
    print((rss - rss_d0) / n)
    print()
    d = math.pow(10, ((rss - rss_d0) / n))
    return d


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

    #plt.scatter(data[:,0], data[:,1])
    #plt.plot(data[:,0], (data[:,0]*coef[0] + coef[1]), '-k')
    #plt.show()

    return coef





environments = ['environment0', 'environment1', 'environment2', 'environment3']



path_loss_table = {
    'environment0': {'ch1': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}},  'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch2': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch3' : {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch4': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch5': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch7': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}}},
    'environment1': {'ch1': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}},  'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch2': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch3' : {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch4': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch5': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch7': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}}},
    'environment2': {'ch1': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}},  'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch2': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch3' : {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch4': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch5': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch7': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}}},
    'environment3': {'ch1': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}},  'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch2': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch3' : {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch4': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch5': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}},
                  'ch7': {'A1': {'los': {}, 'nlos': {}}, 'A2': {'los': {}, 'nlos': {}}, 'A3': {'los': {}, 'nlos': {}},
                          'A4': {'los': {}, 'nlos': {}}, 'A5': {'los': {}, 'nlos': {}}, 'A6': {'los': {}, 'nlos': {}},
                          'A7': {'los': {}, 'nlos': {}}, 'A8': {'los': {}, 'nlos': {}}}}
}

spl = calculate_rss_spline()

for environment in environments:
    print(environment)
    data = {}

    with open('../data_set/' + environment + '/' + 'data.json', 'r') as f:
        data = json.load(f)

    # load walking path data
    path = data['path']
    anchors = data['anchors']
    channels = data['channels']
    print(len(path))

    los_exp = []
    nlos_exp = []
    for channel in channels:
        
        
        for anchor in anchors:
            los = []
            nlos = []
            
            for position in path:

                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    rng = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    
                    #rss = item['rss']
                    rss = spl(item['rss_fp'])
                    if 0 == item['nlos']:
                        los.append([rng, rss])
                    elif 1 == item['nlos']:
                        nlos.append([rng, rss])
        
            los = np.array(los)
            nlos = np.array(nlos)
            
            # calculate path loss exponent
            ploss_los = path_loss_exponent(los)
            ploss_nlos = path_loss_exponent(nlos)

            #print(channel, anchor)
            #print(ploss_los)
            #print(ploss_nlos)
            #print()

            path_loss_table[environment][channel][anchor]['los'] = {'n': ploss_los[0], 'c': ploss_los[1]}
            path_loss_table[environment][channel][anchor]['nlos'] = {'n': ploss_nlos[0], 'c': ploss_nlos[1]}

            los_exp.append(ploss_los[0])
            nlos_exp.append(ploss_nlos[0])

            #path_loss_table[environment][channel]['los'] = {'n': ploss_los[0], 'c': ploss_los[1]}
            #path_loss_table[environment][channel]['nlos'] = {'n': ploss_nlos[0], 'c': ploss_nlos[1]}
        
        print(channel)
    plt.hist(los_exp)
    plt.show()
    plt.hist(nlos_exp)
    plt.show()
            

print(path_loss_table)  

####
# Ranging errors
####
ranging_errors = {}

for environment in environments:
    print(environment)
    data = {}

    with open('../data_set/' + environment + '/' + 'data.json', 'r') as f:
        data = json.load(f)

    # load walking path data
    path = data['path']
    anchors = data['anchors']
    channels = data['channels']
    print(len(path))

    for channel in channels:
        los = []
        nlos = []
        
        for anchor in anchors:
            
            for position in path:

                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    rng = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    
                    NLOS = item['nlos']
                    if 0 == NLOS:
                        NLOS = 'los'
                    else:
                        NLOS = 'nlos'
                    pl = path_loss_table[environment][channel][anchor][NLOS]
                    rng_est = estimate_range(spl(item['rss']), pl['c'], pl['n'])
                    if 0 == item['nlos']:
                        los.append([rng, rng_est])
                    elif 1 == item['nlos']:
                        nlos.append([rng, rng_est])
        
        los = np.array(los)
        nlos = np.array(nlos)

        # detect outliers
        #rng2d = np.concatenate((los[0,:].reshape((-1,1)), los[1,:].reshape((-1,1))), axis=1)
        #rng2d = np.concatenate((los[0,:].reshape((-1,1)), los[1,:].reshape((-1,1))), axis=1)

        print(los)
        print(los.shape)

        # compute DBSCAN
        db = DBSCAN(eps=2.0, min_samples=5).fit(los)
        labels = db.labels_
        n_noise_ = list(labels).count(-1)
        print(n_noise_)

        los = los[db.core_sample_indices_]
            
        plt.scatter(los[:,0], los[:,1])
        plt.show()

        # compute DBSCAN
        db = DBSCAN(eps=2.0, min_samples=5).fit(nlos)
        labels = db.labels_
        n_noise_ = list(labels).count(-1)
        print(n_noise_)

        nlos = nlos[db.core_sample_indices_]

        plt.scatter(nlos[:,0], nlos[:,1])
        plt.show()
            


   
