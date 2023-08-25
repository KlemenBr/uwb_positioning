#!/usr/bin/python3
import numpy as np
import data
import time
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import collections
from scipy import interpolate
import json


    
def calculate_position_ls(anchor_positions, ranges):
    """
    Estimate node position with LS estimation
    :param anchor_positions: numpy array of arrays with anchor positions
    :param ranges: array of ranges to selected anchors
    :return position: estimated x,y position of node
    """

    # get number of anchors
    num_anch = anchor_positions.shape[0]
    if num_anch >= 2:
        A_arr = []
        for i in range(anchor_positions.shape[0]):
            # -2*xi -2*yi 1
            A_arr.append([-2 * float(anchor_positions[i,0]), -2 * float(anchor_positions[i,1]), 1])
        A = np.matrix(A_arr)
        
        # build b matrix
        b_arr = []
        for i in range(anchor_positions.shape[0]):
            # di^2 - xi^2 - yi^2
            b_arr.append([np.power(ranges[i], 2) - np.power(anchor_positions[i,0], 2) - np.power(anchor_positions[i,1], 2)])
        b = np.matrix(b_arr)
        
        if np.linalg.matrix_rank(A_arr) > 2:
            # construct diagonal matrix with reciprocal values of weights array
            position = np.linalg.inv(A.T*A)*A.T*b
            position = np.asarray(position.reshape((1,-1)))[0,:2]
        else:
            position = None
    else:
        position = None
    
    return position


def calculate_position_wls(anchor_positions, ranges, weights):
    """
    Estimate node position with WLS estimation
    :param anchor_positions: numpy array of arrays with anchor positions
    :param ranges: array of ranges to selected anchors
    :param weights: array of estimated ranging errors
    :return position: estimated x,y position of node
    """
    # get number of anchors
    num_anch = anchor_positions.shape[0]
    if num_anch >= 2:
        A_arr = []
        for i in range(anchor_positions.shape[0]):
            # -2*xi -2*yi 1
            A_arr.append([-2 * float(anchor_positions[i,0]), -2 * float(anchor_positions[i,1]), 1])
        A = np.matrix(A_arr)
        
        # build b matrix
        b_arr = []
        for i in range(anchor_positions.shape[0]):
            # di^2 - xi^2 - yi^2
            b_arr.append([np.power(ranges[i], 2) - np.power(anchor_positions[i,0], 2) - np.power(anchor_positions[i,1], 2)])
        b = np.matrix(b_arr)
        
        if np.linalg.matrix_rank(A_arr) > 2:
            # construct diagonal matrix with reciprocal values of weights array
            W = np.diag(weights.reshape((-1)))
            position = np.linalg.inv(A.T*W*A)*A.T*W*b
            position = np.asarray(position.reshape((1,-1)))[0,:2]
        else:
            position = None
    else:
        position = None
    
    return position



channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
environments = ['environment0', 'environment1', 'environment2', 'environment3']


# import data
start_time = time.time()

for environment in environments:

    for channel in channels:
        print("Environment: " + environment)
        print("Channel: " + channel)

        # model name
        model_name = 'model_' + environment + '_' + channel

        ds = data.DataSet_Positioning(channel, environment, model_name)

        walking_path = ds.get_walking_path()
        anchors = ds.get_anchors()

        anchor_positions = ds.get_anchor_positions()

        start_time = time.time()

        stats = []
        stats_compensated = []
        err_sum = 0
        err_sum_comp = 0


        for position in walking_path:
            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
            arr = ds.get_positioning_data(pos_name, 1000)

            estimated_positions = []
            compensated_positions = []

            for item in arr:
                ranges = item[:,0]
                errest = item[:,1] 
                estimated_positions.append(calculate_position_ls(anchor_positions, ranges))
                weights = np.reciprocal(ranges * errest)
                weights = np.clip(weights, a_min=0.01, a_max=100)
                # compensate ranges for estimated ranging error
                ranges = np.subtract(ranges, errest)
                compensated_positions.append(calculate_position_wls(anchor_positions, ranges, weights))
            

            estimated_positions = np.asarray(estimated_positions)  
            compensated_positions = np.asarray(compensated_positions)

            # calculate positoning error
            pos_err = np.sqrt(np.power((estimated_positions[:,0] - float(position['x'])),2) + 
                                np.power((estimated_positions[:,1] - float(position['y'])),2))
            pos_err_compensated = np.sqrt(np.power((compensated_positions[:,0] - float(position['x'])),2) + 
                                np.power((compensated_positions[:,1] - float(position['y'])),2))

            # add mean error to the cumulative error
            err_sum = err_sum + np.mean(pos_err)
            err_sum_comp = err_sum_comp + np.mean(pos_err_compensated)

            stats.append(pos_err)
            stats_compensated.append(pos_err_compensated)

        print(np.mean(stats))
        print(np.mean(stats_compensated))
     
        plt.figure(figsize=(10,6), dpi=300, layout='tight')
        ax1 = plt.subplot(2,1,1)
        ax1.boxplot(stats, sym='')
        ax1.set_title('a) ' + environment + ' ' + channel + '; ' + 'mean error: %0.2f m, ' % np.mean(stats) + 'cumulative mean error: %0.1f m' % err_sum)
        ax1.set_xticks(np.arange(0,len(stats_compensated)+1,step=5))
        ax1.set_xticklabels(np.arange(0,len(stats_compensated)+1,step=5))
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Error [m]')
        plt.grid()

        ax2 = plt.subplot(2,1,2)
        ax2.set_title('b) ' + environment + ' ' + channel + '; ' + 'mean error: %0.2f m, ' % np.mean(stats_compensated) + 'cumulative mean error: %0.1f m' % err_sum_comp)
        ax2.boxplot(stats_compensated, sym='')
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xticks(np.arange(0,len(stats_compensated)+1,step=5))
        ax2.set_xticklabels(np.arange(0,len(stats_compensated)+1,step=5))
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Error [m]')
        plt.grid()
 

        filename = '../data_set/technical_validation/positioning_wls/' + environment + '_' + channel + '.png'
        print('Saving ' + filename)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  
     
        del ds

        print("--- %s ---" % (time.time() - start_time))
        

print("--- %s ---" % (time.time() - start_time))       

