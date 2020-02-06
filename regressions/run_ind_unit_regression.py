import numpy as np
import h5py
import h5py
import numpy as np
import pickle
from hvm_dataset_class import hvm_dataset
from regression_class_v3 import regressionClass
import sys



def main():
    print sys.argv
    unit_id = int(sys.argv[1])  # 0-168 for IT
    region_id = int(sys.argv[2])  # 0 for V4 1 for IT
    model_id = int(sys.argv[3])
    should_time_average = int(sys.argv[4])
    num_splits = 10

    # Different options
    regions = ['V4','IT'] #Which region do you want to predict?
    models = ['alexnet_features_random_filt.hdf5', 'alexnet_features.hdf5'] #Do you want to predict from alexnet or randfilt?
    time_bin_options = [ [80,100,120,140,160,180,200], ['time-averaged']] # Do you want to time average or predict ind time bins?

    region = regions[region_id]
    model = models[model_id]
    time_bins = time_bin_options[should_time_average]

    layers_we_care_about = [4,5,7,9,11,12]
    layers = ['conv1','conv1-relu1','pool1','conv2','conv2-relu2','pool2','conv3','conv3-relu3','conv4','conv4-relu4','conv5','conv5-relu5','pool3']


    if should_time_average == 0:
        mapTo_ = np.load('/home/axsys/es3773/issaLab/data/neuralData/time_bin_'+str(region)+'_data.npy', allow_pickle=True)[()]
    else:
        mapTo_ = np.load('/home/axsys/es3773/issaLab/data/neuralData/'+str(region)+'_data.npy', allow_pickle=True)


    alexnet_features = '/home/axsys/es3773/issaLab/data/alexnetData/'+model


    with h5py.File(alexnet_features, 'r') as f_in:

        for layer_id in layers_we_care_about:
            print layers[layer_id]

            print (5760,np.product(np.shape(f_in['features.'+str(layer_id)])[1:]))
            mapFrom = np.reshape(f_in['features.'+str(layer_id)], (5760, np.product(np.shape(f_in['features.'+str(layer_id)])[1:])))


            for time_bin in time_bins:
                print time_bin
                if time_bin != 'time-averaged':
                    mapTo = mapTo_[time_bin]
                else:
                    mapTo = mapTo_

                regObj = regressionClass(mapFrom[640:,:],mapTo)
                r2 = regObj.run_regression_for_ind_unit( unit_id, num_splits)
                np.save('/home/axsys/es3773/issaLab/data/regressionData/'+model.split('.hdf5')[0]+'_layer_'+layers[layer_id]+'_to_ITbyTimeBin_'+str(time_bin)+'_unit_'+str(unit_id)+'_'+str(num_splits)+'_splits', r2)





if __name__ == '__main__':
    main()