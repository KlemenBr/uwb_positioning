import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

envs = {'environment2': {'path': '../data_set/raw_data/environment2/'}}

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A6', 'A7']

# prepare filenames for the process of range offset calculation
filenames = []
for channel in channels:
    for anchor in anchors:
        filenames.append(channel + '_' + anchor)

data = {}


for environment in envs.keys():

    # load walking path
    walking_path = envs[environment]['path']+'walking_path.csv'
    df = pd.read_csv(walking_path, sep=',', header=None, skiprows=1)
    wp_data = df.values
    
    # load data    
    data['path'] = []
    data['measurements'] = {}
    data['channels'] = channels
    data['anchors'] = anchors


    for position in wp_data:
        pos_name = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # add position to path
        data['path'].append({'x': '%.2f' % position[0], 'y': '%.2f' % position[1], 'z': '%.2f' % position[2], 'name': pos_name})
        # prepare empty dictionary for position's data
        data['measurements'][pos_name] = {}

        # set input folder loaction
        folder_in = envs[environment]['path'] +'data_offset/' + pos_name
        print(folder_in)

        # go through files and load data
        for pair in filenames:
            # get data for one channel and one anchor e.g. 'ch1_A1.csv'
            file = pair + '.csv'
            channel = pair.split('_')[0]
            anchor = pair.split('_')[1]
            filepath_in = folder_in + '/' + file

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


# load walking path data
path = data['path']
channels = data['channels']
anchors = data['anchors']
print(len(path))

for channel in channels:
    rng_error = {}
    for anchor in anchors:
        rng_error[anchor] = []

    for anchor in anchors:
        for position in path:

            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
    
            rerr = []

            for item in data['measurements'][pos_name][anchor][channel]:
                # calculate euclidean distance
                range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                np.power((item['y_anchor'] - item['y_tag']),2) + 
                                np.power((item['z_anchor'] - item['z_tag']),2))
                rerr.append(item['range'] - range)

                
            rng_error[anchor].append(np.array([np.min(rerr), np.mean(rerr), np.max(rerr)]))
        
        rng_error[anchor] = np.asarray(rng_error[anchor])

    plt.figure(figsize=(10,6), layout='tight', dpi=300)
    #plt.title(environment + ' ' + channel + ' ' + 'err_sum: %f' % err_sum + 'err_sum_comp: %f' % err_sum_comp)
    ts = np.arange(int(len(path)))

    plt.subplot(2,1,1)
    plt.title('a) A6 Ranging Error')
    plt.plot(ts, rng_error['A6'][:,1], color='gray', label='Mean Error [m]')
    plt.fill_between(x=ts, y1=rng_error['A6'][:,0], y2=rng_error['A6'][:,2], label='Error [m]', color='blue')
    plt.xlabel('Position')
    plt.ylabel('Error [m]')
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.title('b) A7 Ranging Error')
    plt.plot(ts, rng_error['A7'][:,1], color='gray', label='Mean Error [m]')
    plt.fill_between(x=ts, y1=rng_error['A7'][:,0], y2=rng_error['A7'][:,2], label='Error [m]', color='blue')
    plt.xlabel('Position')
    plt.ylabel('Error [m]')
    plt.grid()
    plt.legend()

    folder_out = '../data_set/technical_validation/range_error_A6/'
    if not os.path.exists(folder_out):
        print('creating empty folder')
        os.makedirs(folder_out)
    filename = folder_out + environment + '_' + channel + '_A6' + '.png'
    print('Saving ' + filename)
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close() 



   
