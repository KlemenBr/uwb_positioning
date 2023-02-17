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
    anchors = data['anchors']
    channels = data['channels']
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

        if (location != 'location2'):
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


        filename = '../data_set/technical_validation/range/' + location + '_' + channel + '.png'
        print('Saving ' + filename)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()  



   
