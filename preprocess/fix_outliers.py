import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


header = 'v1.1\nTAG_ID,ANCHOR_ID,X_TAG,Y_TAG,Z_TAG,X_ANCHOR,Y_ANCHOR,Z_ANCHOR,NLOS,RANGE,FP_INDEX,RSS,RSS_FP,FP_POINT1,FP_POINT2,FP_POINT3,STDEV_NOISE,CIR_POWER,MAX_NOISE,RXPACC,CHANNEL_NUMBER,FRAME_LENGTH,PREAMBLE_LENGTH,BITRATE,PRFR,PREAMBLE_CODE,CIR...'

legend = [
    {'name': 'tag_id', 'index': 0},
    {'name': 'anchor_id', 'index': 1},
    {'name': 'x_tag', 'index': 2},
    {'name': 'y_tag', 'index': 3},
    {'name': 'z_tag', 'index': 4},
    {'name': 'x_anchor', 'index': 5},
    {'name': 'y_anchor', 'index': 6},
    {'name': 'z_anchor', 'index': 7},
    {'name': 'nlos', 'index': 8},
    {'name': 'range', 'index': 9},
    {'name': 'fp_index', 'index': 10},
    {'name': 'rss', 'index': 11},
    {'name': 'rss_fp', 'index': 12},
    {'name': 'fp_point1', 'index': 13},
    {'name': 'fp_point2', 'index': 14},
    {'name': 'fp_point3', 'index': 15},
    {'name': 'stdev_noise', 'index': 16},
    {'name': 'cir_power', 'index': 17},
    {'name': 'max_noise', 'index': 18},
    {'name': 'rxpacc', 'index': 19},
    {'name': 'channel_number', 'index': 20},
    {'name': 'frame_length', 'index': 21},
    {'name': 'preamble_length', 'index': 22},
    {'name': 'bitrate', 'index': 23},
    {'name': 'prfr', 'index': 24},
    {'name': 'preamble_code', 'index': 25},
    {'name': 'cir', 'index': 26}
]

ch_list = [1, 2, 3, 4, 5, 7, 8]
envs = {'location0': {'location': './data_set/raw_data/location0/'},
		'location1': {'location': './data_set/raw_data/location1/'},
		'location2': {'location': './data_set/raw_data/location2/'},
		'location3': {'location': './data_set/raw_data/location3/'}}

channel_names = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH7']


for location in envs.keys():
    print(location)

    # count the outliers
    num_outliers = 0

    # load walking path
    walking_path = envs[location]['location']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=0)
    wp_data = df.values

    # go through positions and fix position data
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # create fixed folder data
        folder_in = envs[location]['location']+'data_position/'+folder
        folder_out = envs[location]['location']+'data_outliers/'+folder
        print(folder_out)
        if not os.path.exists(folder_out):
            print('creating empty folder')
            os.makedirs(folder_out)
    
        # go through files
        for file in os.listdir(folder_in):
            filepath_in = folder_in + '/' + file
            filepath_out = folder_out + '/' + file
            print(filepath_out)
            # load data table
            df = pd.read_csv(filepath_in, sep=',', header=None, skiprows=2)
            data = df.values
            
            # go through table and fix it
            euclidean_ranges = np.sqrt((np.power((data[:,5] - data[:,2]),2) + np.power((data[:,6] - data[:,3]),2) + np.power((data[:,7] - data[:,4]),2)).astype(float))
            ranges = data[:,9]

            # detect outliers
            rng2d = np.concatenate((euclidean_ranges.reshape((-1,1)), ranges.reshape((-1,1))), axis=1)
            # compute DBSCAN
            db = DBSCAN(eps=2.0, min_samples=5).fit(rng2d)
            labels = db.labels_
            n_noise_ = list(labels).count(-1)

            if n_noise_ > 0:
                num_outliers += n_noise_
                #print(np.where(labels==-1))
                junk = ranges[np.where(labels==-1)]
                #print(junk)
                #print(ranges)
                #print()

            data = data[db.core_sample_indices_]

            # Write to output file
            np.savetxt(filepath_out, data, fmt='%s', delimiter=',', header=header)
    print("num_outliers: %u" % num_outliers)


            
    
