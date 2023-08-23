import json
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


envs = {'environment0': {'path': '../data_set/raw_data/environment0/'},
		'environment1': {'path': '../data_set/raw_data/environment1/'},
		'environment2': {'path': '../data_set/raw_data/environment2/'},
		'environment3': {'path': '../data_set/raw_data/environment3/'}}

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

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
        rng = {}
        trng = {}

        # init empty lists for results
        for anchor in anchors:
            rng[anchor] = []
            trng[anchor] = []

        for anchor in anchors:
            for position in path:

                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
    
                rerr = []

                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    rerr.append(item['range'])

                
                rng[anchor].append(np.array([np.min(rerr), np.mean(rerr), np.max(rerr)]))
                trng[anchor].append(range)
        
            rng[anchor] = np.asarray(rng[anchor])
            trng[anchor] = np.asarray(trng[anchor])
        
        

        plt.figure(figsize=(10,18), layout='tight')
        ts = np.arange(int(len(path)))

        plt.subplot(8,1,1)
        plt.title('A1')
        plt.plot(ts, rng['A1'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A1'][:,0], y2=rng['A1'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A1'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,2)
        plt.title('A2')
        plt.plot(ts, rng['A2'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A2'][:,0], y2=rng['A2'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A2'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,3)
        plt.title('A3')
        plt.plot(ts, rng['A3'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A3'][:,0], y2=rng['A3'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A3'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,4)
        plt.title('A4')
        plt.plot(ts, rng['A4'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A4'][:,0], y2=rng['A4'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A4'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,5)
        plt.title('A5')
        plt.plot(ts, rng['A5'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A5'][:,0], y2=rng['A5'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A5'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,6)
        plt.title('A6')
        plt.plot(ts, rng['A6'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A6'][:,0], y2=rng['A6'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A6'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,7)
        plt.title('A7')
        plt.plot(ts, rng['A7'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A7'][:,0], y2=rng['A7'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A7'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        plt.subplot(8,1,8)
        plt.title('A8')
        plt.plot(ts, rng['A8'][:,1], color='white')
        plt.fill_between(x=ts, y1=rng['A8'][:,0], y2=rng['A8'][:,2], label='Measured Range', color='blue')
        plt.plot(ts, trng['A8'], color='black', label='Euclidean Distance')
        plt.xlabel('Position')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.legend()

        filename = '../data_set/technical_validation/range_raw/' + environment + '_' + channel + '.png'
        print('Saving ' + filename)
        plt.savefig(filename, bbox_inches='tight')
        #plt.show()
        plt.close()  



   
