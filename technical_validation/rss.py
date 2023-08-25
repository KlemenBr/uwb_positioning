import json
import numpy as np
import matplotlib.pyplot as plt


environments = ['environment0', 'environment1', 'environment2', 'environment3']


for environment in environments:
    print(environment)
    data = {}

    with open('../data_set/' + environment + '/' + 'data.json', 'r') as f:
        data = json.load(f)

    # load walking path data
    path = data['path']
    anchors = data['anchors']
    channels = data['channels']
    print(len(path))

    los_exp = []
    nlos_exp = []
    for channel in channels:
        
        
        for anchor in anchors:
            los = []
            nlos = []
            
            for position in path:

                pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

                for item in data['measurements'][pos_name][anchor][channel]:
                    # calculate euclidean distance
                    rng = np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2))
                    
                    rss = item['rss']
                    if 0 == item['nlos']:
                        los.append([rng, rss])
                    elif 1 == item['nlos']:
                        nlos.append([rng, rss])
        
            los = np.array(los)
            nlos = np.array(nlos)


            plt.figure(figsize=(14,6), dpi=300, layout='tight')
            plt.title(environment + ' ' + anchor + ' ' + channel)
            #ax = plt.subplot(1,2,1)
            plt.scatter(los[:,0], los[:,1], label='LoS RSS')
            plt.scatter(nlos[:,0], nlos[:,1], label='nLoS RSS')
            #plt.title('a) ')
            plt.xlabel('range [m]')
            plt.ylabel('RSS [dBm]')
            plt.grid()
            plt.legend()

            filename = '../data_set/technical_validation/rss/' + environment + '/' + anchor + '_' + channel + '.png'
            print('Saving ' + filename)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            



            


   
