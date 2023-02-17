#!/usr/bin/python3
"""
This file is intended to test, how the number of steps affect the model performance
"""

import numpy as np
import sys
import time
import math
import argparse
import json
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, LeakyReLU, Conv1D, MaxPooling1D, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow import convert_to_tensor
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import data


# import data
start_time = time.time()

# calculate input data size
pooling_size = 2
input_data_size = int(math.ceil(150 / (math.pow(pooling_size, 2))) * (math.pow(pooling_size, 2)))

ANCHOR_ID = 0
X_TAG = 1
Y_TAG = 2
Z_TAG = 3
X_ANCHOR = 4
Y_ANCHOR = 5
Z_ANCHOR = 6
NLOS = 7
RANGE = 8
FP_INDEX = 9
RSS = 10
RSS_FP = 11
FP_POINT1 = 12
FP_POINT2 = 13
FP_POINT3 = 14
STDEV_NOISE = 15
CIR_POWER = 16
MAX_NOISE = 17
RXPACC = 18
CHANNEL_NUMBER = 19
FRAME_LENGTH = 20
PREAMBLE_LENGTH = 21
BITRATE = 22
PRFR = 23
PREAMBLE_CODE = 24
CIR = 25



L1_patch_w = 2
L1_depth = 10
L2_patch_w = 4
L2_depth = 10
L3_patch_w = 4
L3_depth = 10
L4_patch_w = 4
L4_depth = 20
L5_patch_w = 4
L5_depth = 20
fc_size = 128

def save_model(model, model_name):
	"""
	Saves trained model to disk as JSON for structure and HDF5 for weights
	:param model: Keras model
	:param model_name: model name
	:return:
	"""
	model_json = model.to_json()
	with open('./models/' + str(model_name) + '.json', 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('./models/' + str(model_name) + '.h5')


def generate_model(batch_normalization=True, dropout_regularization=True):
	model = Sequential()
	# L1
	model.add(Conv1D(input_shape=(input_data_size,1,), filters=L1_depth, kernel_size=L1_patch_w, strides=1, padding='valid',
				 use_bias=True, kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L2
	model.add(Conv1D(filters=L2_depth, kernel_size=L2_patch_w, strides=2, padding='same', use_bias=True,
				 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L3
	model.add(Conv1D(filters=L3_depth, kernel_size=L3_patch_w, strides=2, padding='same', use_bias=True,
				 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# pool			 
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
	# L4
	model.add(Conv1D(filters=L4_depth, kernel_size=L4_patch_w, strides=1, padding='valid', use_bias=True,
				 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L5
	model.add(Conv1D(filters=L5_depth, kernel_size=L5_patch_w, strides=2, padding='same', use_bias=True,
				 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# pool
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
	# flatten
	model.add(Flatten())
	# dense
	model.add(Dense(units=fc_size, activation='relu', use_bias=True,
				kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	if  True == dropout_regularization:
		model.add(Dropout(rate=0.5))
	model.add(Dense(units=1, activation='linear', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	
	return model
	

channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
locations = ['location0', 'location1', 'location2', 'location3']


# prepare train data
ds = data.DataSet()

for i in range(len(locations)):
	for channel in channels:
		test_location = locations[i]
		print("Train model to be tested in environment: " + str(test_location))
		print("Channel: " + channel)
		names_mask = []
		for k in range(len(locations)):
			if i != k:
				names_mask.append(locations[k])
		names_mask = np.asarray(names_mask)

		print(names_mask)
		
		# prepare train data
		train_data = []
		train_labels = []
		for j in range(len(locations)-1):
			# load data first
			true_ranges, meas_ranges, labels, cir = ds.load_data_set(channel, names_mask[j])
			train_data.append(cir)
			train_labels.append(labels)
		train_data = np.concatenate(train_data, axis=0)	
		train_labels = np.concatenate(train_labels, axis=0)	

		# load test data
		print("Loading test data: " + test_location)
		true_ranges, meas_ranges, labels, cir = ds.load_data_set(channel, test_location)
		test_data = cir
		test_labels = labels

		print("Train samples: " + str(train_data.shape))
		print("Test samples: " + str(test_data.shape))
		

		model = generate_model(batch_normalization=True, dropout_regularization=True)
		adam = Adam(learning_rate=1e-4)
		model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
		earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min', baseline=None, restore_best_weights=True)
		history = model.fit(x=train_data, y=train_labels, batch_size=8192, epochs=1000, verbose=1, validation_split=0.1, shuffle=True, callbacks=[earlystopping])
		evaluation = model.evaluate(x=test_data, y=test_labels, verbose=0)
		print(evaluation)
		print("mean_squared_error: %0.2f" % evaluation[0])
		fn = 'model_' + test_location + '_' + channel
		save_model(model, fn)
		print("--- %s seconds ---" % (time.time() - start_time))
				
		# clear model
		keras.backend.clear_session()
		del model

