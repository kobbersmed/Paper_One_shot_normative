import numpy as np 
import copy
# from io_funcs import *

def truncate_ts(data_coll, startind, ts_len):
    data_coll_trunc = copy.deepcopy(data_coll)
    for key in data_coll:
        ts = data_coll[key]
        ts_new = ts[startind:ts_len,:]
        data_coll_trunc[key] = ts_new
        
        # if len(ts_new) < ts_len:
        #     warnings.warn(f'Time series has been truncated to length {ts_len-startind}. First point is index {startind} in original time series')
    return data_coll_trunc

def truncate_ts_all(data_coll, remove_firstinds = 1):
    if remove_firstinds:
        # startind, maxstartind_regionind, maxstind_subjID = find_startind_index_all_subj(data_coll)
        startind = 8
    else:
        startind = 0
    T_range = find_tvec_range(data_coll)
    if (T_range[0] != T_range[1]) or (startind!=0):
        data_coll_t = truncate_ts(data_coll, startind, T_range[0])
    else:
        data_coll_t = data_coll

    return data_coll_t

def concat_ts(data_coll_stand):
    """
    Concatenates time series to fit the data format for the GLHMM module.
    Parameters:
    data_coll_stand: Dictionary containing the keys:Subject IDs and values:time series.
    Time series MUST BE truncated to same length.
    Returns:
    ts_concat: Array-like of shape (n_subjects*n_timepoints, n_features). Concatenated time series from subjects.
    T_t: Array-like of shape (n_subjects, 2). Start and end indices of time series for each subject.

    """
    ts_list = [data_coll_stand[key] for key in data_coll_stand.keys()]
    ts_length = ts_list[0].shape[0]
    T_t_list = [[i*ts_length, (i+1)*ts_length] for i in range(len(ts_list))]
    T_t = np.array(T_t_list)
    ts_concat = np.concatenate(ts_list)
    return ts_concat, T_t

def find_startind_index_all_subj(data_coll):
    maxstartind_subjects_list = []
    maxstartind_regioninds_subjects_list = []
    for i in range(len(data_coll)):
        ID = list(data_coll.keys())[i]
        data = data_coll[ID]
        stpoint, stpointregionind = find_startind_subject(data)
        maxstartind_subjects_list.append(stpoint)
        maxstartind_regioninds_subjects_list.append(stpointregionind)
    maxstartind_subjects = max(maxstartind_subjects_list)
    maxstind_subjind = np.argmax(np.array(maxstartind_subjects_list))
    maxstind_subjID = list(data_coll.keys())[maxstind_subjind]
    maxstartind_regionind = maxstartind_regioninds_subjects_list[maxstind_subjind]

    return maxstartind_subjects, maxstartind_regionind, maxstind_subjID

def find_startind_subject(data):
    ####
    #Parameter:
    #   data: array-like of shape [no of timepoints] x [no of regions]
    startpoints = []
    for i in range(data.shape[1]):
        stpoint = find_within1sd_index(data[:,i])
        startpoints.append(stpoint)
    maxstartpoint = max(startpoints)
    maxstpoint_regionind = np.argmax(np.array(startpoints)) 
    return maxstartpoint, maxstpoint_regionind

def find_meancrossing_index(ts_data):
    ctrl = 0
    i = 0
    sign_startpoint = np.sign(ts_data[0] - np.mean(ts_data))

    while not ctrl:
        i += 1
        sign_newstart = np.sign(ts_data[i] - np.mean(ts_data[i:]))
        if sign_newstart != sign_startpoint:
            startind = i
            ctrl = 1

    return startind

def find_within1sd_index(ts_data):
    i = -1
    ctrl = 0
    while not ctrl:
        i += 1
        onesd_int = [np.mean(ts_data[i:]) - np.std(ts_data[i:]), np.mean(ts_data[i:]) + np.std(ts_data[i:])]
        ctrl = (ts_data[i] >onesd_int[0]) and (ts_data[i]<onesd_int[1])
    startind = i

    return startind


def subtract_mean_ts(data_coll):
    data_coll_demaned = copy.deepcopy(data_coll)
    for key in data_coll:
        ts = data_coll[key]
        ts_demean = ts - np.mean(ts,0)
        data_coll_demaned[key] = ts_demean

    return data_coll_demaned

def standardize_ts(data_coll, standard_type):
    #Standardizes each time series to mean 0 and sd 1.
    #standard_type is 0, 1 og 2. 0: No standardization. 1: Removes mean. 2: Removes mean and scales to variance = 1.
    data_coll_standardized = copy.deepcopy(data_coll)

    ctrl = 0
    if standard_type > 0:
        for key in data_coll:
            ctrl += 1
            ts = data_coll[key]
            ts_mean = np.mean(ts,0)
            ts_new = np.zeros_like(ts)
            if standard_type == 2:
                ts_sd = np.std(ts, 0)                
                for i in range(ts.shape[1]):
                    ts_new[:,i] = (ts[:,i]-ts_mean[i])/ts_sd[i]
                if ctrl==len(data_coll):
                    # print('Time series were standardized to mean 0 and variance 1')
                    ctrl = 0
            elif standard_type == 1:
                for i in range(ts.shape[1]):
                    ts_new[:,i] = (ts[:,i]-ts_mean[i])
                if ctrl==len(data_coll):
                    # print('Time series were standardized to mean 0')
                    ctrl = 0

            data_coll_standardized[key] = ts_new
    else:
        data_coll_standardized = data_coll
    return data_coll_standardized

def inverse_standardize_age(age_stand, age_orig, transf_type):
    raise Exception('Inverse age standardization not implemented')
    if transf_type != 0:
        age_range = np.array([np.min(age_orig), np.max(age_orig)])
        age_fact = (age_range[1]-age_range[0])/transf_type
        age_div = age_stand + np.min(age_orig/age_fact) + (transf_type-1)
        age_inv = age_div * age_fact
    else:
        age_inv = age_stand

    return age_inv

def calc_average_Y_acrTime(IDs_list, data_coll):

    n_subj = len(data_coll)
    data_example = data_coll[list(data_coll.keys())[0]]
    p_model = data_example.shape[1]
    Y_ave = np.zeros([n_subj, p_model])
    for i in range(len(IDs_list)):
        key = IDs_list[i]
        ts_all = data_coll[key]
        Y_ave[i, :] = np.mean(ts_all, 0)

    return Y_ave

def gen_average_ts_acrSubj(IDs, data_coll):
    """Generates average time series across individuals at each ROI.
    
    Parameters:
    -----------
    IDs: List of subject IDs (either int or str)
    data_coll: Dict contatining responses for each ID. Response is array-like of shape (n_t, n_parcels)
    
    Returns:
    --------
    ave_ts_matrix: Array-like of shape (no_timepoint, no_parcels). Average time series over subjects specified in IDs
    
    """
    t_len = find_tvec_range(data_coll)[0]
    print('Minimum length of time series is ' + str(t_len) + ' points.')
    n_parcels = np.shape(data_coll[str(IDs[0])])[1]
    matrix_sum = np.zeros([t_len, n_parcels])
    for i in range(len(IDs)):
        matrix_sum += data_coll[str(IDs[i])][:t_len, :]
        
    ave_ts_matrix = matrix_sum/len(IDs)
    return ave_ts_matrix

def find_tvec_range(data_coll):
    min_len = float('inf')
    max_len = float('-inf')
    for key in data_coll:
        len_tmp = data_coll[str(key)].shape[0]
        min_len = min(min_len, len_tmp)
        max_len = max(max_len, len_tmp)

    ts_len_range = (min_len, max_len)
    return ts_len_range


def calc_autocorr_ts(ts, lag=1):
    ###Determines the autocorrelation of a time series 
    #Parameters:
    #   ts: array of length T. Time series vector at equidistant time points
    #   lag: List of integers. Each element is the number of timepoints to shift the time series.   
    # Returns:
    #   ts_ac: 1D array of same length as the input parameter lag. Autocorrelation of input time series for each value of lag

    if np.max(lag) > (len(ts)-2):
        raise Exception("Choice of lag value must be at most the number of timepoints minus 2")

    if type(lag) is int:
        lag = [lag]

    ts_ac = np.zeros(len(lag))
    for i in range(len(lag)):
        ts_ac[i] = np.corrcoef(ts[:-lag[i]], ts[lag[i]:])[0,1]

    return ts_ac

def calc_autocorr_all(data_coll, parcel, lag=1):
    ###Determines the autocorrelation of a time series 
    #Parameters:
    #   data_coll: Dictionary with key:subject ID and value:array-like of shape (no_timepoint, no_parcels)
    #   parcel: Integer. The parcel at which to calculate the autocorrelation.
    #   lag: List of integers. Each element is the number of timepoints to shift the time series.   
    # Returns:
    #   ts_ac_all: Array-like of shape (n_subjects, len(lag)). Autocorrelation of input time series for each value of lag

    if type(lag) is int:
        lag = [lag]

    ts_ac_all = np.zeros([len(data_coll), len(lag)])
 
    k = 0
    for key in data_coll:
        ts = data_coll[key]
        ts_ac_all[k,:] = calc_autocorr_ts(ts[:,parcel], lag)
        k +=1

    return ts_ac_all

def get_autocorr_statistics(data_coll, lag):
    ###Determines the average and sd of autocorrelation of timeseries at all parcels. Mean and sd are across subjects. 
    #Parameters:
    #   data_coll: Dictionary with key:subject ID and value:array-like of shape (no_timepoint, no_parcels)
    #   lag: Int or list or array of integers. Each element is the number of timepoints to shift the time series.   
    # Returns:
    #   ts_ac_all: Array-like of shape (n_subjects, len(lag)). Autocorrelation of input time series for each value of lag

    
    test_data = data_coll[list(data_coll.keys())[0]]
    n_parcels = test_data.shape[1]

    if type(lag) is int:
        lag = [lag]


    ac_ave = np.zeros([n_parcels, len(lag)])
    ac_sd = np.zeros([n_parcels, len(lag)])

    for i in range(n_parcels):
        ac_all = calc_autocorr_all(data_coll, i, lag)
        ac_ave[i, :] = np.mean(ac_all,0)
        ac_sd[i,:] = np.std(ac_all, 0)

    return ac_ave, ac_sd


