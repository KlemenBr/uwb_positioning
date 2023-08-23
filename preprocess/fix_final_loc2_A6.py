import os
import pandas as pd


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
    
    # save fixed data    
    for position in wp_data:
        folder = '%.2f_' % position[0] + '%.2f_' % position[1] + '%.2f' % position[2]
        # create fixed folder data
        folder_in = envs[environment]['path']+'data_offset/'+folder
        folder_out = envs[environment]['path']+'data_final/'+folder
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
            

            if (environment == 'environment2') and (anchor == 'A6'):
                pass
            else:
                print(filepath_out)
                # copy file
                with os.popen('cp ' + filepath_in + ' ' +  filepath_out) as proc:
                    pass
                



    
