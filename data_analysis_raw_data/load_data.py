import os
import pandas as pd
import numpy as np
import time
import json
import pickle


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

#data = {}

for environment in envs.keys():
    print("Loading data for " + environment)

    # load walking path
    walking_path = envs[environment]['environment']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=1)
    wp_data = df.values

    data = {}
    data['path'] = []
    data['measurements'] = {}
    if environment != 'environment2':
        data['anchors'] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    else:
        data['anchors'] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'A8']
    data['channels'] = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']

    for position in wp_data:
        pos_name = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # add position to path
        data['path'].append({'x': '%.2f' % position[0], 'y': '%.2f' % position[1], 'z': '%.2f' % position[2], 'name': pos_name})
        # prepare empty dictionary for position's data
        data['measurements'][pos_name] = {}

        # set input folder loaction
        folder_in = '../data_set/raw_data/' + environment +'/data_offset/' + pos_name
        print(folder_in)

        # go through files and load data
        for pair in filenames:
            # get data for one channel and one anchor e.g. 'ch1_A1.csv'
            file = pair + '.csv'
            channel = pair.split('_')[0]
            anchor = pair.split('_')[1]
            filepath_in = folder_in + '/' + file
            #print(filepath_in)

            if ((environment == 'environment2') and (anchor == 'A6')):
                pass
            else:
                if anchor not in data['measurements'][pos_name].keys():
                    data['measurements'][pos_name][anchor] = {}
                data['measurements'][pos_name][anchor][channel] = []

                # read data
                df = pd.read_csv(filepath_in, sep=',', header=None, skiprows=2)
                loaded_data = df.values
                # go through table and fix it
                for row in loaded_data:
                    '''
                    data structure:
                    data = {
                        path: [{x:, y:, z:, name:}, {x:, y:, z:, name:}...]
                        "anchors": ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                        "channels": ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
                        measurements: {
                            position: {
                                anchor:{
                                    channel: [
                                            [measurement 0],
                                            [measurement 1],
                                            ...
                                            [mesurement N]
                                        ]
                                    }
                                }
                            }
                        }
                    '''

                    # transform row into dictionary
                    measurement = {}
                    for item in legend:
                        if 'cir' == item['name']:
                            measurement['cir'] = [member for member in row[item['index']:]]
                        else: 
                            measurement[item['name']] = row[item['index']]

                    data['measurements'][pos_name][anchor][channel].append(measurement)


    # write data to json file
    print("saving dictionary in .json format...")
    fnm = environment + '.json'
    with open(fnm, 'w') as f:
        f.write(json.dumps(data))

    print("finished")

 