import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


locations = ['location0', 'location1', 'location2', 'location3']

for location in locations:
    print(location)
    data = {}

    with open('../data_set/' + location + '/' + 'data.json', 'r') as f:
        data = json.load(f)

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

                
                rng_error[anchor].append(np.array(rerr))

        plt.figure(figsize=(10,18), layout='tight', dpi=300)
        ts = np.arange(int(len(path)))

        ax = plt.subplot(8,1,1)
        ax.set_title('A1')
        ax.boxplot(rng_error['A1'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        ax = plt.subplot(8,1,2)
        ax.set_title('A2')
        ax.boxplot(rng_error['A2'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        ax = plt.subplot(8,1,3)
        ax.set_title('A3')
        ax.boxplot(rng_error['A3'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        ax = plt.subplot(8,1,4)
        ax.set_title('A4')
        ax.boxplot(rng_error['A4'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        ax = plt.subplot(8,1,5)
        ax.set_title('A5')
        ax.boxplot(rng_error['A5'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        if (location != 'location2'):
            ax = plt.subplot(8,1,6)
            ax.set_title('A6')
            ax.boxplot(rng_error['A6'], sym='')
            ax.set_xticks(np.arange(0,len(path)+1,step=5))
            ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
            ax.set_xlabel('Position')
            ax.set_ylabel('Error [m]')
            plt.grid()

        ax = plt.subplot(8,1,7)
        ax.set_title('A7')
        ax.boxplot(rng_error['A7'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()

        ax = plt.subplot(8,1,8)
        ax.set_title('A8')
        ax.boxplot(rng_error['A8'], sym='')
        ax.set_xticks(np.arange(0,len(path)+1,step=5))
        ax.set_xticklabels(np.arange(0,len(path)+1,step=5))
        ax.set_xlabel('Position')
        ax.set_ylabel('Error [m]')
        plt.grid()


        filename = '../data_set/technical_validation/range_error/' + location + '_' + channel + '.png'
        print('Saving ' + filename)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()  



   
