import os
import pandas as pd
import numpy as np


header = 'v1.1\nTAG_ID,ANCHOR_ID,X_TAG,Y_TAG,Z_TAG,X_ANCHOR,Y_ANCHOR,Z_ANCHOR,NLOS,RANGE,FP_INDEX,RSS,RSS_FP,FP_POINT1,FP_POINT2,FP_POINT3,STDEV_NOISE,CIR_POWER,MAX_NOISE,RXPACC,CHANNEL_NUMBER,FRAME_LENGTH,PREAMBLE_LENGTH,BITRATE,PRFR,PREAMBLE_CODE,CIR...'


ch_list = [1, 2, 3, 4, 5, 7, 8]
envs = {'location0': {'location': './data_set/raw_data/location0/'},
		'location1': {'location': './data_set/raw_data/location1/'},
		'location2': {'location': './data_set/raw_data/location2/'},
		'location3': {'location': './data_set/raw_data/location3/'}}

channel_names = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH7']


for location in envs.keys():
    print(location)

    # load walking path
    walking_path = envs[location]['location']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=0)
    wp_data = df.values

    # go through positions and fix position data
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # go through files and fix tag positions
        # create fixed folder data
        folder_in = envs[location]['location']+'data/'+folder
        folder_out = envs[location]['location']+'data_position/'+folder
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
            for row in data:
                row[2] = '%.2f' % position[0]
                row[3] = '%.2f' % position[1]
                row[4] = '%.2f' % position[2]
                #row = row.astype(str)
            # Write to output file
            np.savetxt(filepath_out, data, fmt='%s', delimiter=',', header=header)


    
