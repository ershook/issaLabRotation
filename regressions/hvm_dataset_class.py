import numpy as np
import h5py
import matplotlib as plt
import scipy.stats as stats
import pickle

class hvm_dataset(object):
    def __init__(self, DATA_PATH = "/share/issa/users/shared/neural_data_hvm/ventral_neural_data.hdf5", verbose=True):
        
        self.DATA_PATH = DATA_PATH
        self.verbose = verbose
        self.Ventral_Dataset = h5py.File(self.DATA_PATH)
        self.getImageMeta()
        self.getNeuralData()
    
    def getImageMeta(self):
        self.image_meta = self.Ventral_Dataset['image_meta']
        self.all_images = self.Ventral_Dataset['images']
        self.unique_objects = np.unique(self.Ventral_Dataset['image_meta']['object_name'])
        self.number_unique_objects = len(self.unique_objects)
            
        self.categories = np.array(self.Ventral_Dataset['image_meta']['category'][:])   #array of category labels for all images  --> shape == (5760,)
        self.unique_categories = np.unique(self.categories)                #array of unique category labels --> shape == (8,)
        self.objects = np.array(self.Ventral_Dataset['image_meta']['object_name'][:])   #array of object labels for all images --> shape == (5670,)
        self.unique_objects = np.unique(self.objects)                      #array of unique object labels --> shape == (64,) 
        
        #now let's use what we just defined to create a dictionary whose keys are categories and whose values
        #are arays of unique objects in each category
        self.objects_by_category = {c: np.unique(self.objects[self.categories == c]) 
                               for c in self.unique_categories}
        self.objects_sorted_by_categories = np.concatenate([self.objects_by_category[c] for c in self.unique_categories])
        
        #we can see that there are three unique values in the variation_level field:
        self.var_level = np.array(self.Ventral_Dataset['image_meta']['variation_level'][:])
        self.unique_var_levels = np.unique(self.var_level)
        
        self.image_counts_by_object_and_varlevel = {(o, v): ((self.var_level == v) & (self.objects == o)).sum() 
                                       for o in self.unique_objects 
                                       for v in ['V0', 'V3', 'V6']}
        
        #let's check the prediction made above 
        self.image_counts_by_category_and_varlevel = {(c, v): ((self.var_level == v) & (self.categories == c)).sum() 
                                         for c in self.unique_categories
                                         for v in ['V0', 'V3', 'V6']}
        
        self.image_object_means_by_category_var0 = np.row_stack([[self.all_images[(self.objects == o) & (self.var_level == 'V0'), :, :].mean(0)
                                                                for o in self.objects_by_category[c]]
                                                                for c in self.unique_categories])

        self.image_object_means_by_category_var6 = np.row_stack([[self.all_images[(self.objects == o) & (self.var_level == 'V6'), :, :].mean(0)
                                                                        for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories])

        self.image_object_means_by_category_var0_flat = self.image_object_means_by_category_var0.reshape((64, 256**2))
        self.image_object_means_by_category_var6_flat = self.image_object_means_by_category_var6.reshape((64, 256**2))
        
     
    def getNeuralData(self):
        self.Neural_Data = self.Ventral_Dataset['time_averaged_trial_averaged']
        print(np.shape(self.Neural_Data))
        self.neural_meta = self.Ventral_Dataset['neural_meta']
        #these are indices into the neurons dimension, defining different subsets of neurons
        #in different brain areas
        self.V4_NEURONS = self.Ventral_Dataset['neural_meta']['V4_NEURONS'][:]
        self.IT_NEURONS = self.Ventral_Dataset['neural_meta']['IT_NEURONS'][:]
        self.V4_Neural_Data = self.Neural_Data[:, self.V4_NEURONS]
        self.IT_Neural_Data = self.Neural_Data[:, self.IT_NEURONS]
        
        if self.verbose:
            print('V4 neural data shape is %d images X %d V4 neurons' % self.V4_Neural_Data.shape)
            print('IT neural data shape is %d images X %d IT neurons' % self.IT_Neural_Data.shape)
        

        self.V4_object_means_by_cat_obj = self._object_means_by_category('V4', cat_obj = True)
        self.V4_object_means_by_category = self._object_means_by_category('V4')
        
        
        self.IT_object_means_by_cat_obj = self._object_means_by_category('IT', cat_obj = True)
        self.IT_object_means_by_category = self._object_means_by_category('IT')
        
        
        self.V4_object_means_by_category_var6 = self._object_means_by_category('V4', var_level='V6')
        self.V4_object_means_by_category_var3 = self._object_means_by_category('V4', var_level='V3')
        self.V4_object_means_by_category_var0 = self._object_means_by_category('V4', var_level='V0')

        
        self.IT_object_means_by_category_var6 = self._object_means_by_category('IT', var_level='V6')
        self.IT_object_means_by_category_var3 = self._object_means_by_category('IT', var_level='V3')
        self.IT_object_means_by_category_var0 = self._object_means_by_category('IT', var_level='V0')
        
        
        
        
        # By monkey 
        self.chabo = self.Ventral_Dataset['neural_meta']['ANIMAL_INFO'][:] == 'Chabo'
        self.chabo_IT_NEURONS = self.IT_NEURONS[self.chabo[self.IT_NEURONS]]
        self.chabo_IT_Neural_Data = self.Neural_Data[:, self.chabo_IT_NEURONS]

        self.chabo_IT_object_means_by_category_var6 = self._object_means_by_category('chabo_IT', var_level='V6')
        
        self.tito = self.Ventral_Dataset['neural_meta']['ANIMAL_INFO'][:] == 'Tito'
        self.tito_IT_NEURONS = self.IT_NEURONS[self.tito[self.IT_NEURONS]]
        self.tito_IT_Neural_Data = self.Neural_Data[:, self.tito_IT_NEURONS]

        self.tito_IT_object_means_by_category_var6 = self._object_means_by_category('tito_IT', var_level='V6')
        
        
        
        self.time_binned_trial_averaged = self.Ventral_Dataset['time_binned_trial_averaged']
        
        self.sorted_IT_Neural_Data_means = np.array([self.IT_Neural_Data[(self.var_level == v) & (self.objects == o)].mean(0)
                                        for c in self.unique_categories
                                        for v in self.unique_var_levels 
                                        for o in self.objects_by_category[c]])
        
        
        self.IT_Data_by_trial = self._IT_Data_by_trial()
        self.V4_Data_by_trial = self._V4_Data_by_trial()
        
        self.by_trial_IT_Neural_Data_objmeans_sorted_by_category = {}
        for vl in self.unique_var_levels:
            level_number = vl[-1]
            arr = self.Ventral_Dataset['time_averaged']['variation_level_%s' % level_number][:, :, self.IT_NEURONS]
            objects_at_var_level = self.objects[self.var_level == vl]
            arr1 = np.array([arr[:, objects_at_var_level == o].mean(1) for o in self.objects_sorted_by_categories])
            self.by_trial_IT_Neural_Data_objmeans_sorted_by_category[vl] = arr1
        
        self._get_data_by_time_bin()
        self._get_v3_v6_data()

    def _get_v3_v6_data(self):
    
        IT_data_V3 = self.V4_Data_by_trial['V3'][:,:46,:]
        IT_data_V6 = self.V4_Data_by_trial['V6'][:,:46,:]
        V4_data_V3 = self.IT_Data_by_trial['V3'][:,:46,:]
        V4_data_V6 = self.IT_Data_by_trial['V6'][:,:46,:]

        IT_data = np.vstack((IT_data_V3, IT_data_V6))
        V4_data = np.vstack((V4_data_V3, V4_data_V6))
        self.IT_V36 = np.swapaxes(IT_data,1,2) #Data dimensions are num_images x num_units x num_trial
        self.V4_V36 = np.swapaxes(V4_data,1,2) 


    def _get_data_by_time_bin(self):
        #Want dictionary with keys of time-bin that returns tensor of stacked v3+v6 images of shape images, units, trials
        self.time_bin_data_IT = {}
        self.time_bin_data_V4 = {}
        
        for interval in self.Ventral_Dataset['/time_binned'].keys():
            v3_data = np.array(self.Ventral_Dataset['/time_binned'][interval]['variation_level_3']).swapaxes(0, 1).swapaxes(1,2)[:,:,:46]
            v6_data = np.array(self.Ventral_Dataset['/time_binned'][interval]['variation_level_6']).swapaxes(0, 1).swapaxes(1,2)[:,:,:46]
            data = np.concatenate((v3_data, v6_data), axis=0)
            self.time_bin_data_IT[int(interval[:-2])] = data[:,self.IT_NEURONS,:]
            self.time_bin_data_V4[int(interval[:-2])] = data[:,self.V4_NEURONS,:]
            
        
    def _object_means_by_category(self, area, cat_obj = False, var_level = None):
        """ Helper method for computing object means by category both at the category level and object level"""
        
        if var_level == None:
        
            if area == 'IT':
                object_means_by_category = [[self.IT_Neural_Data[self.objects == o].mean(0) for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories]

            elif  area == 'V4':
                object_means_by_category = [[self.V4_Neural_Data[self.objects == o].mean(0) for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories]
        else:
            print(area, cat_obj, var_level)
            if area == 'IT':
            
                object_means_by_category = [[self.IT_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                        for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories]
            elif area == 'V4':
                object_means_by_category = [[self.V4_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                        for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories]
            elif area == 'chabo_V4':
                object_means_by_category = [[self.chabo_V4_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                            for o in self.objects_by_category[c]]
                                                                            for c in self.unique_categories]
            
            elif area == 'chabo_IT':
                object_means_by_category = [[self.chabo_IT_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                            for o in self.objects_by_category[c]]
                                                                            for c in self.unique_categories]
                
            elif area == 'tito_IT': 
                
                object_means_by_category = [[self.tito_IT_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                for o in self.objects_by_category[c]]
                                                                for c in self.unique_categories]
            elif area == 'tito_V4': 
                
                object_means_by_category = [[self.tito_V4_Neural_Data[(self.objects == o) & (self.var_level == var_level)].mean(0) 
                                                                        for o in self.objects_by_category[c]]
                                                                        for c in self.unique_categories]
            
        if cat_obj:
            return np.array(object_means_by_category)
        
        else:
            return np.row_stack(object_means_by_category)
        
        
    def zscored_over_images(features):
        if len(features.shape) == 2:
            features = stats.zscore(features, axis=0)
        return features
    
        
    def _IT_Data_by_trial(self):
        """
        IT data time averaged but not trial averaged, sorted by variation level. 
        Datastructure: Dict
        Keys: ['V0', 'V3', 'V6']
        Values: Tensor of num_images x num_trials x num_units
        
        """
        IT_Data_by_trial = {}
        for vl in self.unique_var_levels:
            level_number = vl[-1]
            arr = self.Ventral_Dataset['time_averaged']['variation_level_%s' % level_number][:, :, self.IT_NEURONS]
            IT_Data_by_trial[vl] = arr.swapaxes(0, 1)
        return IT_Data_by_trial

    def _V4_Data_by_trial(self):
        """
        IT data time averaged but not trial averaged, sorted by variation level. 
        Datastructure: Dict
        Keys: ['V0', 'V3', 'V6']
        Values: Tensor of num_images x num_trials x num_units
        
        """
        V4_Data_by_trial = {}
        for vl in self.unique_var_levels:
            level_number = vl[-1]
            arr = self.Ventral_Dataset['time_averaged']['variation_level_%s' % level_number][:, :, self.V4_NEURONS]
            V4_Data_by_trial[vl] = arr.swapaxes(0, 1)
        return V4_Data_by_trial

    def RDM(self, data_matrix, do_corr_rows = True, metric = 'corr'):
      
        if not do_corr_rows:
            data_matrix = data_matrix.T
        if metric == 'corr':
            return 1 - np.corrcoef(data_matrix)
        elif metric == 'euclidean':
            pass

    #useful utility function for plotting HDF5 dimension labels
    def dimnames(self, dataset):
        dims = dataset.dims  #get the dimension object
        dimlist = [x.label for x in dims.keys()]  #get the label attribute
        dimlist = map(str, dimlist)  #cast everything to string instead of "unicode" ... complicated rathole ... not strictly necessary
        return dimlist

    def closeHDF5(self):
        self.Ventral_Dataset.close()
        
        
    def plot_various_trial_analyses(self,neuron_ind, var_level):
        plt.figure(figsize=(16, 5))

        #the first thing we want to do is just plot the data average
        #so first get the data for all trials
        neuron_i_data_by_trial = self.by_trial_IT_Neural_Data_objmeans_sorted_by_category[var_level][:, :, neuron_ind]
        #now take the mean over the second dimension -- the trial dimension
        neuron_i_data_trial_mean = neuron_i_data_by_trial.mean(1)
        #for convenience, let's compute the min and max values of the neural response
        minval = neuron_i_data_trial_mean.min()
        maxval = neuron_i_data_trial_mean.max()
        #now let's plot the responses across objects
        plt.plot(neuron_i_data_trial_mean)
        #and block stuff to make the categories easier to see
        plt.fill_between(np.arange(64), minval, maxval, 
                         where=(np.arange(64) / 8) % 2, color='k', alpha=0.2)
        plt.xticks(np.arange(0, 64, 8) + 4, self.unique_categories, rotation=30);
        plt.ylabel('Neural Response of neuron %d' % neuron_ind)
        plt.ylim(minval, maxval)
        plt.xlabel('Responses for Variation %s images' % var_level)

        #now let's look at two trials -- the first and 6th ones, for example 
        t1 = 0; t2 = 5
        t1_data = neuron_i_data_by_trial[:, t1]
        t2_data = neuron_i_data_by_trial[:, t2]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t1_data)
        plt.xticks(np.arange(0, 64, 8), self.unique_categories, rotation=30);
        plt.title('Neuron %d, trial %d, var %s' % (neuron_ind, t1, var_level))
        plt.subplot(1, 2, 2)
        plt.plot(t2_data)
        plt.xticks(np.arange(0, 64, 8), self.unique_categories, rotation=30);
        plt.title('Neuron %d, trial %d, var %s' % (neuron_ind, t2, var_level))

        #let's do a scatter plot of the responses to one trial vs the other
        plt.figure()
        plt.scatter(t1_data, t2_data)
        plt.xlabel('responses of neuron %d, trial %d, %s'% (neuron_ind, t1, var_level))
        plt.ylabel('responses of neuron %d, trial %d, %s'% (neuron_ind, t2, var_level))

        #how correlated are they exactly between trials? let's use pearson correlation
        rval = stats.pearsonr(t1_data, t2_data)[0]
        plt.title('Correlation for varlevel %s images = %.3f' % (var_level, rval))

        #in fact, let's have a look at the correlation for all pairs of trials 
        fig = plt.figure(figsize = (7, 7))
        #the numpy corrcoef function basically gets the pairwise pearson correlation efficiently
        corrs = np.corrcoef(neuron_i_data_by_trial.T)
        #now let's plot the matrix of correlations using the matshow function
        plt.colorbar(fig.gca().matshow(corrs))
        plt.xlabel('trials of neuron %d' % neuron_ind)
        plt.ylabel('trials of neuron %d' % neuron_ind)
        plt.title('Between-trial correlations for varlevel %s' % var_level)


    #this makes us curious to look at standard deviations of responses across trials
    def plot_trial_avg_data_with_stds(self, neuron_ind, var_level):

        neuron_i_data_by_trial = self.by_trial_IT_Neural_Data_objmeans_sorted_by_category[var_level][:, :, neuron_ind]
        neuron_i_data_trial_mean = neuron_i_data_by_trial.mean(1)
        neuron_i_data_trial_std = neuron_i_data_by_trial.std(1)
        minval = neuron_i_data_trial_mean.min()
        maxval = neuron_i_data_trial_mean.max()
        plt.plot(neuron_i_data_trial_mean)
        plt.fill_between(np.arange(64), minval, maxval, 
                         where=(np.arange(64) / 8) % 2, color='k', alpha=0.2)

        plt.fill_between(np.arange(64), 
                         neuron_i_data_trial_mean - neuron_i_data_trial_std,
                         neuron_i_data_trial_mean + neuron_i_data_trial_std,
                         color='b', alpha=0.2)

        plt.xticks(np.arange(0, 64, 8) + 4, self.unique_categories, rotation=30);
        plt.ylabel('Neural Responses')
        plt.ylim(minval, maxval)
        plt.title('Responses for neuron %d Variation %s images' % (neuron_ind, var_level))
        plt.xlim(0, 64)
        
        
    #here's a very simple implementation of split-half reliability
    #this is a NON-boostrapping version

    def get_correlation(self, data_by_trial, num_trials, num_splits):
        """arguments:
              data_by_trial -- (numpy array) the data
                 assumes a tensor with structure is (stimuli, trials)

              num_trials -- (nonnegative integer) how many trials to consider

              num_splits (nonnegative integer) how many splits of the data to make

           returns:
              array of length num_splits
        """

        
        #get total number of trials
        num_total_trials = data_by_trial.shape[1]

        #you better not ask for more trials than you actually have
        assert num_trials <= num_total_trials, "You asked for %d trials but there's only %d" % (num_trials, num_total_trials)

        #we want to make sure that while we select groups of trials basically randomly,
        #that we can still exactly reproduce our results later
        #so to do this, we use a constructed random number generator to select trial groups
        #and seed the generator with seed = 0 (could be any non-negative integer, but the seed
        #*must* be set for this to be reproducible
        random_number_generator = np.random.RandomState(seed=0)

        corrvals = []
        for split_index in range(num_splits):
            #construct a new permutation of the trial indices
            perm =  random_number_generator.permutation(num_total_trials)

            #take the first num_trials/2 and second num_trials/2 pieces of the data
            first_half_of_trial_indices = perm[:num_trials / 2]
            second_half_of_trial_indices = perm[num_trials / 2: num_trials]

            #mean over trial dimension
            mean_first_half_of_trials = data_by_trial[:, first_half_of_trial_indices].mean(axis=1)
            mean_second_half_of_trials = data_by_trial[:, second_half_of_trial_indices].mean(axis=1)

            #compute the correlation between the means
            corrval = stats.pearsonr(mean_first_half_of_trials, 
                                     mean_second_half_of_trials)[0]
            #add to the list
            corrvals.append(corrval)

        return np.array(corrvals)
    
    def getAlexNetFeats(self, layer, raw = False):
        cnnFeats_path  = "/Users/ericashook/Documents/IssaLab/project_0_regressions/cnnFeats/model_***_300PCAs_py2.pickle"
        if raw:
            return np.array(pickle.load( open( cnnFeats_path.replace('***', layer), "rb" )))
        return np.array(pickle.load( open( cnnFeats_path.replace('***', layer), "rb" ) ))[0]
        
