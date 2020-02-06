import numpy as np
import h5py
import matplotlib as plt
#%pylab inline
import scipy.stats as stats
import scipy.stats as ss
import pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import Ridge




class regressionClass:
    
    """
    A class used to perform linear regressions 
    for exampel from features of a neural network to neural data

    ...

    Attributes
    ----------
    mapTo : np tensor of dimension num_images x num_neurons x num_trials
        Y if mapping is Wx=Y
    mapFrom : np tensor of dimension num_images x num_features
        X if mapping is Wx=Y
    image_train_fraction : float
        fraction of total images to train on
    image_test_fraction : float
        fraction of total images to test on
    trial_split_fraction : float
        for split-half on trials what percentage of images should be in each split
    num_image_folds : int
        number of independent train/test image splits should run 
    num_trial_folds : int
        number of independent trial splits should be run
    num_images : int
        x dimension of mapTo; the number of images in mapTo
    num_neurons : int
        y dimension in mapTo; the number of neurons/features
    num_trials : int
        z dimension of mapTo; the number of trials the images where presented
    num_features : int
        y dimension of mapFrom; the number of features of the neural network
    
    Methods
    -------
    fit_model
        fits a linear model ie Wx=Y where x=mapFrom and y=mapTo
    evaluate_model
        evaluate the model on data
    compute_R_column_wise
        compute pearson r column wise
    compute_r_corrected_consis
        compute the corrected consistency measure
    spearman_brown_correction
        apply spearman brown correction
    zscored_over_images
        zscore across images
    get_trial_splits
        determine random indices of each trial split
    get_mapTo_splits
        take the average across the trial split to get mapTo splits of data
    split_mapTo_into_train_test
        split the mapTo trial splits into train/test sets
    get_splits
        parent function that calls et_trial_splits, get_mapTo_splits, split_mapTo_into_train_test, and get_image_train_test_indices
    get_image_train_test_indices
        determine random indices of each train/test splits
    run_regression
        master function that gets splits, fits model, and evaluates the model
    
        
    """

    def __init__(self, mapFrom, mapTo, image_train_fraction = 0.8, seed = 0):
        
        self.mapTo = mapTo 
        self.mapFrom = mapFrom 
        self.train_fraction = image_train_fraction 
        self.trial_split_fraction = .5 
        self.num_images,  self.num_neurons, self.num_trials = np.shape(mapTo)
        self.num_features = np.shape(mapFrom)[1]
        self.rng = np.random.RandomState(seed)



    def get_train_test_indices(self):
        indices = self.rng.permutation(self.num_images)
        train_indices = indices[:int(self.train_fraction * self.num_images)]
        test_indices = indices[int(self.train_fraction * self.num_images):]

        assert list(set(train_indices).intersection(test_indices)) == [] 

        return train_indices, test_indices
    

    def get_trial_splits(self):
        # Get mapTo split across trials
        trial_indices = self.rng.permutation(self.num_trials)
        trial_split_1_indices = trial_indices[:int(self.num_trials * self.trial_split_fraction)] 
        trial_split_2_indices = trial_indices[int(self.num_trials * self.trial_split_fraction):] 
            
        assert list(set(trial_split_1_indices).intersection(trial_split_2_indices)) == []
            
        return trial_split_1_indices, trial_split_2_indices 

    def demean_columns_of_matrix(self, matrix): 
        # demean each col
        return matrix - np.mean(matrix, axis=0)[None,:]

    
    def get_new_lower_boundary(self, current_lower_boundary):
        return current_lower_boundary-10, current_lower_boundary

    def get_new_upper_boundary(self, current_upper_boundary):
        return current_upper_boundary, current_upper_boundary+10


    def fit_ridgeRegressLOOCV(self, X, Y, possible_alphas=np.logspace(-10, 10, 50)):
        """
          X: (n_samples, n_regressors) -- e.g., (n_stim, n_features)
          Y: (n_samples, n_targets) -- e.g., (n_stim, n_voxels)
    
        """
        lower_boundary = -10
        upper_boundary = 10

        clf = RidgeCV(alphas=possible_alphas)
        clf.fit(X,Y)
        alpha = clf.alpha_
        ii_ = 0

        while alpha == lower_boundary or alpha == upper_boundary: 
            print 'hit boundary! for the '+str(ii_)+' time'

            if alpha == lower_boundary: 
                lower_boundary, upper_boundary = self.get_new_lower_boundary(lower_boundary)
            elif alpha == upper_boundary: 
                lower_boundary, upper_boundary = self.get_new_upper_boundary(upper_boundary)
            clf = RidgeCV(alphas= np.logspace(lower_boundary, upper_boundary, 50))
            clf.fit(X,Y)
            alpha = clf.alpha_
            print alpha
            ii_+=1

        return clf, alpha

    def fit_ridgeRegressGivenParam(self, X, Y, alpha):
        reg = Ridge(alpha=alpha)
        reg.fit(X, Y)
        return reg

    def run_regression_for_ind_unit(self, unit_id, num_train_test_splits):

        # Default parameters
        n_alphas = 50; alpha0 = -10; alpha1 = 10
        possible_alphas_ = np.logspace(alpha0, alpha1, n_alphas)


        mapFrom = self.mapFrom
        mapTo_indUnit = np.squeeze(self.mapTo[:,unit_id,:]) #Matrix of shape num_images x num_trials

        results_dict = {}
        results_dict[unit_id] = {}

        for train_test_split in range(num_train_test_splits):

            print train_test_split
            split_id = 'split_'+str(train_test_split)
            results_dict[unit_id][split_id] = {}

            print 'step 0'
            # Get the train/test and split indices -- note we are seeding this so the splits should be the same across units
            train_indices, test_indices = self.get_train_test_indices()
            trial_split_1_indices, trial_split_2_indices = self.get_trial_splits()

            print 'step 1'
            # Get the regressor matrix: In the case of a neural network this is train/test_images x activations 
            # Also demean each column of this regressor matrix --> this implements ridge with a non-isotropic prior on each unit's learned coefficient (see kell et. al. 2018)
            mapFrom_train = self.demean_columns_of_matrix(self.mapFrom[train_indices, :])
            mapFrom_test = self.demean_columns_of_matrix(self.mapFrom[test_indices, :])

            print 'step 2'
            # Here we are taking the presumably neural data and getting the train/test splits and then averaging across the trial splits
            # Note don't think we need to worry about z-scoring because fitting to individual unit 
            mapTo_indUnit_train_split_1 = mapTo_indUnit[train_indices,:][:,trial_split_1_indices].mean(1)
            y_test_split_1 = mapTo_indUnit[test_indices,:][:,trial_split_1_indices].mean(1)

            mapTo_indUnit_train_split_2 = mapTo_indUnit[train_indices,:][:,trial_split_2_indices].mean(1)
            y_test_split_2 = mapTo_indUnit[test_indices,:][:,trial_split_2_indices].mean(1)

            reg_1, alpha = self.fit_ridgeRegressLOOCV(mapFrom_train, mapTo_indUnit_train_split_1)
            reg_2 = self.fit_ridgeRegressGivenParam(mapFrom_train, mapTo_indUnit_train_split_2, alpha) #Want to fit using the chosen alpha

            y_hat_split_1 = reg_1.predict(mapFrom_test)
            y_hat_split_2 = reg_2.predict(mapFrom_test)

            print 'step 3'
            #Compute r(mapFrom, mapTo): Consistency between predictions and true values for split 1 on the ~test~ images
            r_mapFrom_mapTo_per_site = stats.pearsonr(y_test_split_1, y_hat_split_1)[0]
            
            print 'step 4'
            # Compute r(mapTo, mapTo): Consistency between split 1 and split 2 of the mapTo data on the ~test~ images
            r_mapTo_mapTo_per_site = stats.pearsonr(y_test_split_1, y_test_split_2)[0]

            print 'step 5'
            # Compute r(mapFrom, mapFrom): Consistency between y_hat_split_1 and y_hat_split_2 on ~test~ images and compute r between predictions 
            r_mapFrom_mapFrom_per_site = stats.pearsonr(y_hat_split_1, y_hat_split_2)[0]
        
            results_dict[unit_id][split_id]['alpha'] = alpha 
            results_dict[unit_id][split_id]['r_mapFrom_mapTo_per_site'] = r_mapFrom_mapTo_per_site
            results_dict[unit_id][split_id]['r_mapTo_mapTo_per_site'] = r_mapTo_mapTo_per_site
            results_dict[unit_id][split_id]['r_mapFrom_mapFrom_per_site'] = r_mapFrom_mapFrom_per_site

        return results_dict

    