import numpy as np
import pickle


############################################
### I/O FUNCTIONS THAT NEED MODIFICATION ###
############################################


### Functions for preparing running FUNCOIN scripts

def prepare_run_funcoin(**kwargs):

    # Called in funcoin_runscripts.py->funcoin_runnew_splittest_par()
    # Returns:
    #   X_dat: Numpy array of shape (n_subj,q), where the first column is ones (intercept), and the other columns contain the covariates. In our case column 1,2, and 3 contains sex, age, and sex*age for the n_subj subjects.
    #   Y_dat: List of length n_subj containing time series data for each subject. Each element of the list is numpy array of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
    #          The time series data should be standardized to mean 0 and variance 1. 
    #   IDs_list: List of subject IDs. Can be empty, since this value is not used in subsequent analysis.

    X_dat = float('nan') #PROVIDE X MATRIX OF TRAINING DATA HERE
    Y_dat = float('nan') #PROVIDE LIST WITH TIME SERIES DATA HERE


    IDs_list = []

    return X_dat, Y_dat, IDs_list

def funcoin_prepare_from_IDslist(**kwargs):
    # Called in funcoin_runscripts.py->run_funcoin_method() and funcoin_runscripts.py->funcoin_run_normative_UKB_diag()

    # Returns:
    #   X_dat: Numpy array of shape (n_subj,q), where the first column is ones (intercept), and the other columns contain the covariates. In our case column 1,2, and 3 contains sex, age, and sex*age for the n_subj subjects.
    #           The age variable needs to be standardized to the interval [0,1]. To reproduce the standardization from the paper, use the function io_funcs -> standardize_age(age_var, 1)
    #   Y_dat: List of length n_subj containing time series data for each subject. Each element of the list is numpy array of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
    #          The time series data should be standardized to mean 0 and variance 1. 

    X_dat = float('nan') #PROVIDE X MATRIX OF TRAINING DATA HERE
    Y_dat = float('nan') #PROVIDE LIST WITH TIME SERIES DATA HERE

    return X_dat, Y_dat

####Behavioural data

def read_behavioural_csv(data_type = '', **kwargs):
    # Called in funcoin_runscripts.py->funcoin_runnew_splittest_par()
    # Parameter:
    #   data_type: String indicating what data to be loaded. This is specified in the scripts calling the function, where either training, testing, or specific diagnosis data is loaded. 
    # Returns:
    #   covar_vals: Numpy array of shape (n_subj, n_covariates), i.e. (n_subj, q-1). 
    #               For reproducing the results from the paper, the shape should be (n_subj,2) 
    #               where the two columns are sex (coded as 0 for female and 1 for male) and age (cronological age in years)

    if data_type == 'training': # LOAD TRAINING DATA HERE
        covar_vals = float('nan')
    elif data_type == 'testing': # LOAD OUT-OF-SAMPLE TEST DATA HERE
        covar_vals = float('nan')
    elif data_type == 'BP': # LOAD DATA FOR SUBJECTS WITH BIPOLAR DISORDER HERE
        covar_vals = float('nan')
    elif data_type == 'PD': # LOAD DATA FOR SUBJECTS WITH PARKINSONS DISEASE HERE
        covar_vals = float('nan')
    elif data_type == 'MS': # LOAD DATA FOR SUBJECTS WITH MULTIPLE SCLEROSIS HERE
        covar_vals = float('nan')
                        
    return covar_vals



###Time series data

def load_data_stand_calc_FC(data_type = '', **kwargs):
    # Called in funcoin_runscripts.py->funcoin_runnew_splittest_par()
    # Returns:
    #   FC_mat_all: Numpy array of shape (n_subj, n_edges). Each row, i, contains all unique edge values (i.e. the upper triangular excluding the diagonal) from the correlation matrix of subject i.

    if data_type == 'training': # LOAD TRAINING DATA HERE
        FC_mat_all = float('nan')
    elif data_type == 'testing': # LOAD OUT-OF-SAMPLE TEST DATA HERE
        FC_mat_all = float('nan')
    elif data_type == 'BP': # LOAD DATA FOR SUBJECTS WITH BIPOLAR DISORDER HERE
        FC_mat_all = float('nan')
    elif data_type == 'PD': # LOAD DATA FOR SUBJECTS WITH PARKINSONS DISEASE HERE
        FC_mat_all = float('nan')
    elif data_type == 'MS': # LOAD DATA FOR SUBJECTS WITH MULTIPLE SCLEROSIS HERE
        FC_mat_all = float('nan')

    return FC_mat_all






###################################################
### I/O FUNCTIONS THAT DO NOT NEED MODIFICATION ###
###################################################


def standardize_age(age_var, transf_type, dataset):
    #Transforms the age variable linearly to the interval [-1,1]
    #Paramters:
    #   age_var: Array of age values
    #   transf_type: 0, 1, or 2. 0: No transformation. 1: Linearly to [0,1]. 2: Linearly to [-1,1]
    #   dataset: The dataset to be used. In this version the minimum and meximum age are specified, which makes the dataset variable obsolete.
    #Returns:
    #   age_transf: Array of transformed age values. Same length as age_var.

    age_min = 44.56449771689495
    age_max = 85.42865296803666

    age_range = np.array([age_min, age_max])

    if np.ndim(age_var) == 2:
        age_var = np.squeeze(age_var)
    if transf_type != 0:
        age_fact = (age_range[1]-age_range[0])/transf_type
        age_div = age_var/age_fact
        age_trans = age_div-(age_min/age_fact)-(transf_type-1)
    else:
        age_trans = age_var
        
    return age_trans


# Load and save result files

def save_data_pickle(var_names, variabs, filename):
    #Save data file 
    #Parameters: 
    #   var_names: List of strings indicating variable names
    #   variabs: List of the variables to be saved
    #   filename: Str. Specifies the path and name of the file to be saved.
    #Returns:
    #   Saves the data file at the specified location.

    if len(var_names) != len(variabs):
        raise Exception('Warning. Lists of variable names and variables have different lengths')
    
    results_dic = {}
    for i in range(len(var_names)):
        name_string = str(var_names[i])
        variab = variabs[i]
        results_dic[name_string] = variab
    
    with open(filename, 'wb') as handle:
        pickle.dump(results_dic, handle)

def load_data_pickle(filename):
    #Load data file
    #Parameters: 
    #   filename: Str. Specifies the file to be loaded.
    #Returns:
    #   var_names: List of strings indicating variable names
    #   var_dic: Dictionary containing the variable names from var_names as keys and the variable values and values.
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    if type(data) == dict:
        var_names = list(data.keys())
        var_dic = data
    elif (type(data) == list) and (len(data)==2) and (len(data[0])==len(data[1])):
        var_names = data[0]
        variabs = data[1]
        var_dic = {}
        for i in range(len(var_names)):
            name = var_names[i]
            var_dic[name] = variabs[i]
    else:
        raise Exception('Loaded data variable of incompatible format.')

    return var_names, var_dic

# ID list functions:

def get_new_IDs_healthy(**kwargs):

    # This functions used to return a list of healthy subjects' IDs while excluding IDs specified in the input variable IDs_old.
    # For replicating paper results, where the reader provide the data themselves, this function can be ignored. 

    IDs_healthy_new = []

    return IDs_healthy_new

def get_diagnosis_subjects(**kwargs):
    
    # This functions used to return a list of IDs for suvjects with diagnoses specified in an input variable.
    # For replicating paper results, where the reader provide the data themselves, this function can be ignored. 

    diaggroup_IDs = []

    return diaggroup_IDs
