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

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
environments = ['environment0', 'environment1', 'environment2', 'environment3']


for environment in environments:
    data0 = {}
    data1 = {}

    with open(environment + '_unfixed_range' + '.json', 'r') as f:
        data0 = json.load(f)

    with open(environment + '.json', 'r') as f:
        data1 = json.load(f)

    # load walking path data
    path = data0['path']



    for channel in channels:
        for anchor in anchors:
            if ('A6' == anchor) & ('environment3' == environment):
                pass
            else:
                rng_error_los0 = [[],[],[]]
                rng_error_nlos0 = [[],[],[]]
                rng_error_los1 = [[],[],[]]
                rng_error_nlos1 = [[],[],[]]

                for position in path:
                    pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

                    for item in data0['measurements'][pos_name][anchor][channel]:
                        # calculate euclidean distance
                        range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                        np.power((item['y_anchor'] - item['y_tag']),2) + 
                                        np.power((item['z_anchor'] - item['z_tag']),2))
                        range_error = item['range'] - range 

                        #if ('A1' == anchor) & ('ch2' == channel):
                        #    print(pos_name + ': ' + str(item['range']))
                        if -1 > item['range']:
                            print(pos_name + ': ' + str(item['range']))

                        if 'NLOS' == item['nlos']:
                            rng_error_nlos0[0].append(range)
                            rng_error_nlos0[1].append(range_error)
                            rng_error_nlos0[2].append(item['range'])
                        else:
                            rng_error_los0[0].append(range)
                            rng_error_los0[1].append(range_error)
                            rng_error_los0[2].append(item['range'])

                    for item in data1['measurements'][pos_name][anchor][channel]:
                        # calculate euclidean distance
                        range = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                        np.power((item['y_anchor'] - item['y_tag']),2) + 
                                        np.power((item['z_anchor'] - item['z_tag']),2))
                        range_error =  item['range'] - range


                        if 1 == item['nlos']:
                            rng_error_nlos1[0].append(range)
                            rng_error_nlos1[1].append(range_error)
                            rng_error_nlos1[2].append(item['range'])
                        else:
                            rng_error_los1[0].append(range)
                            rng_error_los1[1].append(range_error)
                            rng_error_los1[2].append(item['range'])
                
                plt.figure(figsize=(21,12), tight_layout=True)
                ax = plt.subplot(4,1,1)
                #plt.title('range')
                ax.scatter(rng_error_los0[0], rng_error_los0[2], label='los_unfixed')
                ax.scatter(rng_error_nlos0[0], rng_error_nlos0[2], label='nlos_unfixed')
                ax.grid()
                ax.set_facecolor('cyan')
                ax.legend()

                ax = plt.subplot(4,1,2)
                ax.scatter(rng_error_los1[0], rng_error_los1[2], label='los_fixed')
                ax.scatter(rng_error_nlos1[0], rng_error_nlos1[2], label='nlos_fixed')
                ax.grid()
                ax.set_facecolor('cyan')
                ax.legend()

                ax = plt.subplot(4,1,3)
                #plt.title('ranging error')
                ax.scatter(rng_error_los0[0], rng_error_los0[1], label='los_unfixed')
                ax.scatter(rng_error_nlos0[0], rng_error_nlos0[1], label='nlos_unfixed')
                ax.grid()
                ax.set_facecolor('cyan')
                ax.legend()

                ax = plt.subplot(4,1,4)
                ax.scatter(rng_error_los1[0], rng_error_los1[1], label='los_fixed')
                ax.scatter(rng_error_nlos1[0], rng_error_nlos1[1], label='nlos_fixed')
                ax.grid()
                ax.set_facecolor('cyan')
                ax.legend()

                filename = './figures/' + environment + '/' +  anchor + '_' + channel + '.png'
                print(filename)
                plt.savefig(filename)
                plt.close()
                

