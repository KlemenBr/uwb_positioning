import json
from scipy import interpolate
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

CIRLEN = 152
OFFSET = 5


#channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
#anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
locations = ['location0', 'location1', 'location2', 'location3']


def init_rss_spline():
    """
    This function generates spline representation of actual RSS vs. estimated RSS
    :return: interpolate.spline object
    """

    x = [-150, -149, -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133,
     -132, -131, -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
     -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102.3, -102, -100.5, -99, -98, -97,
     -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86.5, -85.7, -85, -84.5, -83.5, -83, -82.5, -82, -81.5,
     -81, -80.8, -80.5, -80.2, -80.1, -79.9, -79.8, -79.6, -79.5, -79.3, -79.2, -78.9, -78.73, -78.56, -78.39,
      -78.21, -78.04, -77.87, -77.70, -77.53, -77.36, -77.19, -77.01, -76.84, -76.67, -76.50, -76.33, -76.16, -75.98]
    y = range(-150, -48, 1)

    spl = interpolate.interp1d(x, y, kind='cubic', fill_value=(-150, -48), bounds_error=False)

    return spl



def interpolate_rss(rss, tck_spline):
    """
    This function interpolates estimated RSS value with actual RSS vs. estimated RSS according to graph in documentation
    :param rss: measured RSS
    :param tck_spline: spline representation of actual vs. estimated rss
    :return: interpolated RSS
    """
    #interp_rss = interpolate.splev(rss, tck_spline)
    interp_rss = tck_spline(rss)

    return interp_rss


def cpx2abs(complexlist):
	""" convert numpy array of complex number strings to a numpy array of absolute values"""
	temp = np.empty((len(complexlist)), dtype=complex)
	for i in range(temp.shape[0]):
		temp[i] = complex(complexlist[i])
	abs_data = np.absolute(temp)

	return abs_data


def generate_cir(data):
    """
    Find useful CIR index limits inside the input data
    """
    rxpacc = data['rxpacc']
    fp_index = int(data['fp_index'])
    startidx = fp_index - OFFSET
    abs_data = (cpx2abs(data['cir'])/float(rxpacc))[startidx:startidx+CIRLEN]
    abs_data = np.asarray(abs_data).astype(np.float32)
    abs_data = np.expand_dims(abs_data, axis=1)

    return abs_data


def generate_pdp_interpolated(data, tck_spline):
    """
    Find useful PDP index limits inside the input data
    """
    rxpacc = data['rxpacc']
    fp_index = int(data['fp_index'])
    startidx = fp_index - OFFSET
    abs_data = cpx2abs(data['cir'])[startidx:startidx+CIRLEN]
    pdp = 10 * np.log10((abs_data * np.power(2,17))/np.power(data['rxpacc'],2)) - 121.74
    pdp = interpolate_rss(pdp, tck_spline)
    pdp = np.asarray(pdp).astype(np.float32)
    pdp = np.expand_dims(pdp, axis=1)

    return pdp


def generate_pdp(data):
    """
    Find useful PDP index limits inside the input data
    """
    rxpacc = data['rxpacc']
    fp_index = int(data['fp_index'])
    startidx = fp_index - OFFSET
    abs_data = cpx2abs(data['cir'])[startidx:startidx+CIRLEN]
    pdp = 10 * np.log10((abs_data * np.power(2,17))/np.power(data['rxpacc'],2)) - 121.74
    pdp = np.asarray(pdp).astype(np.float32)
    pdp = np.expand_dims(pdp, axis=1)

    return pdp


def load_model(model_name):
    """
    loads model from files
    :param model_name: model name without the .json and .h5 file extensions
    :return: keras classification model
    """
    json_file = open('./models/' + str(model_name) + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('./models/' + str(model_name) + '.h5')
    adam = Adam(learning_rate=1e-4)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    
    return model



class DataSet(object):
    def __init__(self):
        
        pass

    
    def load_data_set(self, channel, location):
        # load data from selected location
        with open('./data_set/' + location + '/' + 'data.json', 'r') as f:
            data = json.load(f)
        
        # load walking path and anchors in a selected data set
        path = data['path']
        anchors = data['anchors']
        channels = data['channels']

        #data_set = []
        true_rng = []
        rng = []
        rng_error = []
        cir = []

        for position in path:
            # define position name string/key
            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

            # go through all anchors
            for anchor in anchors:
                # skip loading data for anchor A6 in location location2
                if ((location == 'location2') and (anchor == 'A6')):
                    pass
                else:
                    for item in data['measurements'][pos_name][anchor][channel]:
                        # generate pdp from raw data
                        cir.append(generate_cir(item))
                        # measured range
                        rng.append(item['range'])
                        # ground truth range
                        # calculate euclidean distance
                        true_rng.append(np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                        np.power((item['y_anchor'] - item['y_tag']),2) + 
                                        np.power((item['z_anchor'] - item['z_tag']),2)))
                        # range error
                        rng_error.append(abs(rng[-1] - true_rng[-1]))

        true_rng = np.asarray(true_rng)
        cir = np.asarray(cir)
        rng = np.asarray(rng)
        rng_error = np.asarray(rng_error)

        del data

        return true_rng, rng, rng_error, cir


class DataSet_Positioning(object):
    def __init__(self, channel, location, error_estimation_model):
        self.channel = channel
        self.location = location
        self.model = load_model(error_estimation_model)

        # private variables
        self._true_rng = []
        self._pos_name = []
        self._position = []
        self._rng = []
        self._rng_error = []
        self._cir = []
        self._estimated_error = []
        self._wpath = []
        self._anchor_position = []
        self._anchor = []
        self._channels = []
        self._anchors = []

        # anchor reference positions
        self._reference_positions = {}

        # generated positioning data set
        self._positioning_array = {}

        self.data_set_init()

    def get_data(self):
        return self._positioning_array
    
    def get_walking_path(self):
        return self._wpath

    def get_anchors(self):
        return self._anchors
    
    def get_channels(self):
        return self._channels
    
    def get_anchor_positions(self):
        apos = []
        for anchor in self._anchors:
            apos.append(self._reference_positions[anchor])
        apos = np.asarray(apos)
        
        return apos
    
    def get_positioning_data(self, posname, niter):
        self._reference_positions = np.asarray(self._reference_positions)
        rng_combinations = []
        
        for i in range(niter):
            arng = []
            for anchor in self._anchors:
                index = np.random.choice(self._positioning_array[posname][anchor]['ranges'].shape[0], 1, replace=False)[0]
                arng.append(self._positioning_array[posname][anchor]['ranges'][index])
            rng_combinations.append(np.asarray(arng))

        rng_combinations = np.asarray(rng_combinations)  

        return rng_combinations
    
    def data_set_init(self):
        # load data from selected location
        with open('./data_set/' + self.location + '/' + 'data.json', 'r') as f:
            data = json.load(f)
        
        # load walking path
        self._wpath = data['path']
        self._channels = data['channels']
        self._anchors = data['anchors']

        for position in self._wpath:
            # define position name string/key
            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']

            # go through all anchors
            for anchor in self._anchors:
                for item in data['measurements'][pos_name][anchor][self.channel]:
                    self._pos_name.append(pos_name)
                    self._position.append(position)
                    self._anchor_position.append(np.array([item['x_anchor'], item['y_anchor']]))
                    self._reference_positions[anchor] = np.asarray([item['x_anchor'], item['y_anchor']])
                    self._anchor.append(anchor)
                    # generate pdp from raw data
                    self._cir.append(generate_cir(item))
                    # measured range
                    self._rng.append(item['range'])
                    # ground truth range
                    # calculate euclidean distance
                    self._true_rng.append(np.sqrt(np.power((item['x_anchor'] - item['x_tag']),2) + 
                                    np.power((item['y_anchor'] - item['y_tag']),2) + 
                                    np.power((item['z_anchor'] - item['z_tag']),2)))
                    # range error
                    self._rng_error.append(abs(self._rng[-1] - self._true_rng[-1]))

        self._true_rng = np.asarray(self._true_rng)
        self._cir = np.asarray(self._cir)
        self._rng = np.asarray(self._rng)
        self._rng_error = np.asarray(self._rng_error)
        self._pos_name = np.asarray(self._pos_name)
        self._position = np.asarray(self._position)
        self._anchor_position = np.asarray(self._anchor_position)
        self._anchor = np.asarray(self._anchor)
        # estimate all errors
        self._estimated_error = self.model.predict(x=self._cir, verbose=0, batch_size=4096)

        for position in self._wpath:
            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
            self._positioning_array[pos_name] = {}
            for anchor in self._anchors:
                self._positioning_array[pos_name][anchor] = {}
                self._positioning_array[pos_name][anchor]['position'] = []
                self._positioning_array[pos_name][anchor]['ranges'] = []

            # create anchor positions and ranges arrays
            #
            # anchor_positions = {
            #    'A1': ['x_anchor', 'y_anchor],
            #    'A2': ['x_anchor', 'y_anchor],
            #    'A3': ['x_anchor', 'y_anchor],
            #    'A4': ['x_anchor', 'y_anchor],
            #    'A5': ['x_anchor', 'y_anchor],
            #    'A6': ['x_anchor', 'y_anchor],
            #    'A7': ['x_anchor', 'y_anchor],
            #    'A8': ['x_anchor', 'y_anchor]}
            # 
            # ranges = {
            #    'A1': [[rng0, rng_error_estimate0] , ... ],
            #    'A2': [[rng0, rng_error_estimate0] , ... ],
            #    'A3': [[rng0, rng_error_estimate0] , ... ],
            #    'A4': [[rng0, rng_error_estimate0] , ... ],
            #    'A5': [[rng0, rng_error_estimate0] , ... ],
            #    'A6': [[rng0, rng_error_estimate0] , ... ],
            #    'A7': [[rng0, rng_error_estimate0] , ... ],
            #    'A8': [[rng0, rng_error_estimate0] , ... ]}
            # 

            #

        for i in range(self._rng.shape[0]):
            pos_name = self._pos_name[i]
            anchor = self._anchor[i]
            self._positioning_array[pos_name][anchor]['ranges'].append(np.asarray([self._rng[i], self._estimated_error[i][0]]))
            self._positioning_array[pos_name][anchor]['position'] = self._anchor_position[i]

        for position in self._wpath:
            pos_name = position['x'] + '_' + position['y'] + '_' + position['z']
            for anchor in self._anchors:
                 self._positioning_array[pos_name][anchor]['ranges'] = np.asarray(self._positioning_array[pos_name][anchor]['ranges'])       
            
        del data

