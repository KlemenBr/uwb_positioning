import os
import pandas as pd
import numpy as np


header = 'v1.1\nTAG_ID,ANCHOR_ID,X_TAG,Y_TAG,Z_TAG,X_ANCHOR,Y_ANCHOR,Z_ANCHOR,NLOS,RANGE,FP_INDEX,RSS,RSS_FP,FP_POINT1,FP_POINT2,FP_POINT3,STDEV_NOISE,CIR_POWER,MAX_NOISE,RXPACC,CHANNEL_NUMBER,FRAME_LENGTH,PREAMBLE_LENGTH,BITRATE,PRFR,PREAMBLE_CODE,CIR...'


envs = {'environment0': {'path': './data_set/raw_data/environment0/'},
		'environment1': {'path': './data_set/raw_data/environment1/'},
		'environment2': {'path': './data_set/raw_data/environment2/'},
		'environment3': {'path': './data_set/raw_data/environment3/'}}

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

# prepare filenames for the process of range offset calculation
filenames = []
for channel in channels:
    for anchor in anchors:
        filenames.append(channel + '_' + anchor)


for environment in envs.keys():

    # load walking path
    walking_path = envs[environment]['path']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=1)
    wp_data = df.values

    distance_table = {'A1': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A2': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A3': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A4': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A5': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A6': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A7': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []},
                    'A8': {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': [], 'ch5': [], 'ch7': []}}

    offset_table = {'A1': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A2': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A3': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A4': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A5': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A6': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A7': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0},
                'A8': {'ch1': 0, 'ch2': 0, 'ch3': 0, 'ch4': 0, 'ch5': 0, 'ch7': 0}}

    # create LoS ground truth range table
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        folder_in = envs[environment]['path']+'data_nlos/'+folder
        print(folder_in)
    
        # go through files
        for pair in filenames:
            # get data for one channel and one anchor e.g. 'ch1_A1.csv'
            file = pair + '.csv'
            channel = pair.split('_')[0]
            anchor = pair.split('_')[1]

            filepath_in = folder_in + '/' + file
            # read data
            df = pd.read_csv(filepath_in, sep=',', header=None, skiprows=2)
            data = df.values
            # check if LOS
            if 1 != data[0,8]:
                # calculate ground truth and mean range
                tag_pos = np.array([data[0,2], data[0,3], data[0,4]])
                anch_pos = np.array([data[0,5], data[0,6], data[0,7]])
                ground_truth_range = np.linalg.norm(tag_pos - anch_pos)
                mean_range = np.mean(data[:,9])
                # add element to distance table
                distance_table[anchor][channel].append([ground_truth_range, mean_range])

    # populate offset table
    for anchor in anchors:
        for channel in channels:
            # get min range and calculate range offset
            temp = np.asarray(distance_table[anchor][channel])
            minimal_range = temp[np.argmin(temp[:,0])]
            offset = minimal_range[1] - minimal_range[0]
            print(offset)
            offset_table[anchor][channel] = offset
            print()

    print(offset_table)

    print("##################################")
    print("Correct data for identified offset")
    print("##################################")
    print()
    
    # save fixed data    
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # create fixed folder data
        folder_in = envs[environment]['path']+'data_nlos/'+folder
        folder_out = envs[environment]['path']+'data_offset/'+folder
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
                range = row[9] - offset_table[anchor][channel]
                row[9] = '%.2f' % range
                row = row.astype(str)

            # Write to output file
            np.savetxt(filepath_out, data, fmt='%s', delimiter=',', header=header)  



    
