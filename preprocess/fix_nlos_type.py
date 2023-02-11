import os
import pandas as pd
import numpy as np


header = 'v1.1\nTAG_ID,ANCHOR_ID,X_TAG,Y_TAG,Z_TAG,X_ANCHOR,Y_ANCHOR,Z_ANCHOR,NLOS,RANGE,FP_INDEX,RSS,RSS_FP,FP_POINT1,FP_POINT2,FP_POINT3,STDEV_NOISE,CIR_POWER,MAX_NOISE,RXPACC,CHANNEL_NUMBER,FRAME_LENGTH,PREAMBLE_LENGTH,BITRATE,PRFR,PREAMBLE_CODE,CIR...'


envs = {'location0': {'location': './data_set/raw_data/location0/'},
		'location1': {'location': './data_set/raw_data/location1/'},
		'location2': {'location': './data_set/raw_data/location2/'},
		'location3': {'location': './data_set/raw_data/location3/'}}

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

# prepare filenames for the process of range offset calculation
filenames = []
for channel in channels:
    for anchor in anchors:
        filenames.append(channel + '_' + anchor)


for location in envs.keys():
    # load walking path
    walking_path = envs[location]['location']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=0)
    wp_data = df.values


    print("##################################")
    print("Correct NLOS data type")
    print("##################################")
    print()
   
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # go through files and fix NLOS data
        # create fixed folder data
        folder_in = envs[location]['location']+'data_outliers/'+folder
        folder_out = envs[location]['location']+'data_nlos/'+folder
        print(folder_out)
        if not os.path.exists(folder_out):
            print('creating empty folder')
            os.makedirs(folder_out)

        # go through files
        for pair in filenames:
            # get data for one channel and one anchor e.g. 'ch1_A1.csv'
            file = pair + '.csv'
            channel = pair.split('_')[0]
            anchor = pair.split('_')[1]
            filepath_in = folder_in + '/' + file
            filepath_out = folder_out + '/' + file
            print(filepath_out)

            # read data
            df = pd.read_csv(filepath_in, sep=',', header=None, skiprows=2)
            data = df.values
            # go through table and fix it
            for row in data:
                nlos = None
                if 'NLOS' == row[8]:
                    nlos = 1
                else:
                    nlos = 0
                row[8] = '%1d' % nlos
                row = row.astype(str)
            # Write to output file
            np.savetxt(filepath_out, data, fmt='%s', delimiter=',', header=header)  
