import numpy as np
from funcoin import Funcoin as fcn
import time
from funcoin_normative_funcs import sort_subjectinds_oneyearagegroups
import os
from io_funcs import read_behavioural_csv, get_new_IDs_healthy, save_data_pickle, load_data_pickle, standardize_age, get_diagnosis_subjects, funcoin_prepare_from_IDslist, prepare_run_funcoin
from multiprocessing import Pool
import warnings
from funcoin_splittest_singlerun import funcoin_splittest_singlerun_func
import platform

def run_funcoin_method(model_params, IDs_list = [], parc_list = [], filename_suf = ''):

    """ Runs the FUNCOIN decomposition with the specified parameters and saves the data in the folder specified by datpath.
        Parameters:
        -----------
        model_params: Dictionary of model parameters. This can be loaded from the function get_model_parameters(model_version) found in current_model_parameters.py
        IDs_list: List of IDs to be used for training the model. If empty, a list of n_subj training IDs will be loaded.
        parc_list: List or numpy array. Specifies which ROIS/parcels/components of the parcellation to be used. If empty, all are used.
        filename_suf: String to be attached to the end of the filename of the result file

        Returns:
        --------
        Saves the model fitting results in a file in the folder specified by the string datpath. The following variables are saved: gamma_mat, beta_mat, IDs_list, X_dat, dfd_vals_geom, parc_list, u_train
    """

    seed_bootstrap = 213    

    n_init = model_params['n_init']
    n_subj = model_params['n_subj']
    sex = model_params['sex']
    site = model_params['site']
    n_parc = model_params['n_parc']
    n_dir = model_params['n_dir']
    covars_list = model_params['covars_list']
    interaction_io = model_params['interaction_io']
    age_squared_io = model_params['age_squared_io'] 
    dataset = model_params['dataset'] 
    betaLinReg = model_params['betaLinReg']
    datpath = model_params['datpath_remote']
    bootstrapCI_io = model_params['bootstrapCI_io']
    n_bootstrap = model_params['n_bootstrap']
    CI_lvl = model_params['CI_lvl']

    rand_init = 1
    ts_stand_type = model_params['ts_stand_type']
    age_transf_type = model_params['age_transf_type']
    seed_initial = model_params['seed_initial']


    filename = funcoin_generate_filename(model_params, 0, 0, filename_suf = filename_suf, n_dir_used=0, datpath=datpath) #Generate filename for result file 
                
    checkfile = os.path.isfile(filename)

    print(f'SAVEFILENAME: {filename}')

    if not checkfile: #Only run code if there is not already a saved result file with the same name

        if len(IDs_list) == 0: 
            IDs_list = get_new_IDs_healthy(n_subj=n_subj, IDs_old = [], sex=sex, site=site, brainhealth_only=True)
            
        X_dat, Y_dat = funcoin_prepare_from_IDslist(IDs_list, covars_list, n_parc, ts_stand_type = ts_stand_type, age_transf_type = age_transf_type, interaction_io=interaction_io, age_squared_io=age_squared_io, dataset=dataset, parc_list=parc_list)

        fcn_obj = fcn()

        if not bootstrapCI_io:
            timeA = time.time()
            fcn_obj.decompose(Y_dat, X_dat, max_comps=n_dir, rand_init = rand_init, n_init = n_init, seed_initial=seed_initial, betaLinReg=betaLinReg)
            decomp_time = time.time()-timeA
            print(f'FUNCOIN decomposition run took {decomp_time}')
        elif bootstrapCI_io:
            timeA = time.time()
            fcn_obj.decompose_bootstrap(Y_dat, X_dat, n_samples=n_bootstrap, max_comps=n_dir, CI_lvl=CI_lvl, rand_init = rand_init, n_init = n_init, seed_initial=seed_initial, betaLinReg=betaLinReg, seed_bootstrap=seed_bootstrap)
       
            decomp_time = time.time()-timeA
            print(f'FUNCOIN decomposition run took {decomp_time}')            

        beta_mat = fcn_obj.beta
        gamma_mat = fcn_obj.gamma
        dfd_vals_geom = fcn_obj.dfd_values_training
        u_train = fcn_obj.u_training

        variabs = [gamma_mat, beta_mat, IDs_list, X_dat, dfd_vals_geom, parc_list, u_train]
        var_names = ['gamma_mat', 'beta_mat', 'IDs_list', 'X_dat', 'dfd_vals_geom', 'parc_list', 'u_train']

        if bootstrapCI_io:
            betas_bootstrap = fcn_obj.betas_bootstrap
            CI_lvl = fcn_obj.CI_lvl_bootstrap
            beta_CI = fcn_obj.beta_CI_bootstrap
            variabs_extra = [CI_lvl, betas_bootstrap, beta_CI]
            var_names_extra = ['CI_lvl', 'betas_bootstrap', 'beta_CI']
            variabs.extend(variabs_extra)
            var_names.extend(var_names_extra)

        save_data_pickle(var_names, variabs, filename)

        return decomp_time


def funcoin_run_normative_UKB_diag(model_params, n_dir_used = 2, IDs_used = [], parc_list = [], filename_suf = '', ddof_val = 0):
    
    """ Loads the results from the FUNCOIN decomposition (performed by calling run_funcoin_method). 
        Thereafter loads out-of-sample and diagnosis IDs, calculates u values and Z-scores as well as several other measures to be used in analysis and visualization of the normative model.
    #        Parameters:
        -----------
        model_params: Dictionary of model parameters. This can be loaded from the function get_model_parameters(model_version) found in current_model_parameters.py
        n_dir_used: The number of components to be used for the normative model.
        IDs_used: List of IDs to be excluded when loading out-of-sample healthy subjects, e.g. the subjects used for training the model. 
        parc_list: List or numpy array. Specifies which ROIS/parcels/components of the parcellation to be used. If empty, all are used.
        filename_suf: String to be attached to the end of the filename of the result file
        ddof_val: The delta degrees of freedom to be used when calculating the standard deviation in the model. 0 is the population covariance (biased), 1 is the (unbiased/centered) sample covariance.

        Returns:
        --------
        Saves various analysis results in a file in the folder specified by the string datpath. The following variables are saved: 
        [IDs_list_train, IDs_outsample, IDs_diags, X_dat_train, X_dat_outsample, X_dats_diag, u_train, u_outsample, u_diags, age_orig_train, age_orig_outsample, age_orig_diag, X_dat_model_sex0, X_dat_model_sex1,  
        R2_in, R2_out, R2s_diags, sex_train, sex_outsample, sex_diag, Zscores_outsample, Zscores_diag,  diag_codes, dfd_vals_geom_train, x_agegroups, x_agegroups_transf, y_pred_Z_sex0, y_pred_Z_sex1, u_groupmeans_sex0, 
        u_groupmeans_sex1, u_groupstds_sex0, u_groupstds_sex1, age_orig_train, age_orig_outsample, age_orig_diag, agegroups_inds_train, agegroups_inds_outsample, agegroups_inds_diag, u_residuals_train, modelpred_train, 
        modelpred_outsample, modelpred_diag, MSE_in, MSE_out, MSEs_diags, u_residuals_outsample, u_residuals_diag, gamma_mat_train, beta_mat_train, set_seed, parc_list]
    """

    timeA = time.time()
    n_subj = model_params['n_subj']
    sex = model_params['sex']
    site = model_params['site']
    n_parc = model_params['n_parc']
    covars_list = model_params['covars_list']
    interaction_io = model_params['interaction_io']
    age_squared_io = model_params['age_squared_io'] 
    dataset = model_params['dataset'] 
    datpath = model_params['datpath_remote']
    n_outsample = model_params['n_outsample']
    diag_codes = model_params['diag_codes']
    diag_labels = model_params['diag_labels']


    commonstd_io = model_params['commonstd_io']
    ts_stand_type = model_params['ts_stand_type']
    age_transf_type = model_params['age_transf_type']
    datpath = model_params['datpath_remote']
    n_outsample_sets = model_params['n_outsample_sets']
    set_seed = model_params['set_seed_norm']

    n_diags = len(diag_codes)

    filename = funcoin_generate_filename(model_params, 0, 0, filename_suf = filename_suf, n_dir_used=n_dir_used, datpath=datpath) #Filename for loading results from decomposition
    savefilename = funcoin_generate_filename(model_params, 1, 0, filename_suf = filename_suf, n_dir_used=n_dir_used, datpath=datpath) #Filename for saving the normative model results



    checkfile = os.path.isfile(filename)

    checksavefile = os.path.isfile(savefilename)
    if checksavefile:
        warnings.warn('Save file already found. Normative modelling already done')
        return []

    if not checkfile:
        print('Did not find the following file:')
        print(filename)
        raise Exception('Please run CAP method before normative modelling')
    else:
        var_names, var_dic = load_data_pickle(filename)


    gamma_mat_train = var_dic['gamma_mat']
    beta_mat_train = var_dic['beta_mat']
    IDs_list_train = var_dic['IDs_list']
    X_dat_train = var_dic['X_dat']
    dfd_vals_geom_train = var_dic['dfd_vals_geom']

    fcn_obj = fcn(gamma = gamma_mat_train, beta = beta_mat_train)

    u_train = var_dic['u_train']

    IDs_used_here = IDs_list_train + IDs_used

    IDs_diags_init = get_diagnosis_subjects(diag_codes, dataset=dataset)

    IDs_diags = []
    if len(IDs_used)>0:
        for i2 in range(len(IDs_diags_init)):
            IDs_diags_here = [IDs_diags_init[i2][i3] for i3 in range(len(IDs_diags_init[i2])) if IDs_diags_init[i2][i3] not in IDs_used_here]
            IDs_diags.append(IDs_diags_here)
    else:
        IDs_diags = IDs_diags_init


    X_dats_diag = []
    Y_dats_diag = []
    u_diags = []
    for k in range(len(IDs_diags)):
        X_dat_here, Y_dat_here = funcoin_prepare_from_IDslist(IDs_diags[k], covars_list, n_parc, ts_stand_type, age_transf_type, interaction_io, age_squared_io, dataset, parc_list=parc_list)
        X_dats_diag.append(X_dat_here)
        Y_dats_diag.append(Y_dat_here)
        u_vals_diag = fcn_obj.transform_timeseries(Y_dat_here)
        u_diags.append(u_vals_diag)
        IDs_used_here.extend(IDs_diags[k])


    #Get healthy outsample IDs without diag subjects:
    n_out_total = n_outsample*n_outsample_sets
    IDs_outsample = get_new_IDs_healthy(n_subj=n_out_total, IDs_old=IDs_used_here, sex=sex, site=site, brainhealth_only=True, random_io=True, set_seed = set_seed, dataset=dataset)

    X_dat_outsample, Y_dat_outsample = funcoin_prepare_from_IDslist(IDs_outsample, covars_list, n_parc, ts_stand_type, age_transf_type, interaction_io, age_squared_io, dataset, parc_list=parc_list)


    u_outsample = fcn_obj.transform_timeseries(Y_dat_outsample)

    R2_in = fcn_obj.score(X_dat_train, u_train[:,:])
    MSE_in = fcn_obj.score(X_dat_train, u_train[:,:], score_type = 'mean_squared_error')
    print(f'Insample R2: {R2_in}. Insample MSE: {MSE_in}')
    
    R2_out = fcn_obj.score(X_dat_outsample, u_outsample[:,:])
    MSE_out = fcn_obj.score(X_dat_outsample, u_outsample[:,:], score_type = 'mean_squared_error')
    print(f'Outsample score: {R2_out}. Average outsample MSE: {MSE_out}')

    R2s_diags = [fcn_obj.score(X_dats_diag[i][:,:], u_diags[i]) for i in range(len(X_dats_diag))]
    MSEs_diags = [fcn_obj.score(X_dats_diag[i][:,:], u_diags[i], score_type = 'mean_squared_error') for i in range(len(X_dats_diag))]
    
    ###AGE GROUPS
    sex_covarind = covars_list.index('sex')
    age_covarind = covars_list.index('age')

    sex_age_train = read_behavioural_csv(data_type = 'training', IDs_list=IDs_list_train, covars = ['sex', 'age'], dataset=dataset)

    sex_age_outsample = read_behavioural_csv(data_type = 'testing', IDs_list=IDs_outsample, covars = ['sex', 'age'], dataset=dataset)


    sex_age_diag = []
    for i in range(len(IDs_diags)):
        diag_label = diag_labels[i]
        sexage_here = read_behavioural_csv(data_type = diag_label, IDs_list=IDs_diags[i], covars = ['sex', 'age'], dataset=dataset)
        sex_age_diag.append(sexage_here)

    sex_train = sex_age_train[:,0]
    age_orig_train = sex_age_train[:,1]
    sex_outsample = sex_age_outsample[:,0]
    age_orig_outsample = sex_age_outsample[:,1]

    sex_diag = [sex_age_diag[i][:,0] for i in range(len(IDs_diags))]
    age_orig_diag = [sex_age_diag[i][:,1] for i in range(len(IDs_diags))]

    age_diag_mins = [np.min(age_orig_diag[i]) for i in range(len(age_orig_diag))]
    age_diag_maxs = [np.max(age_orig_diag[i]) for i in range(len(age_orig_diag))]

    age_orig_outsample_min = np.min(age_orig_outsample) 
    age_orig_outsample_max = np.max(age_orig_outsample) 

    age_min = np.min([np.min(age_orig_train), np.min(age_orig_outsample_min), np.min(age_diag_mins)])
    int_min = int(np.floor(age_min))
    age_max = np.max([np.max(age_orig_train), np.max(age_orig_outsample_max), np.max(age_diag_maxs)])
    int_max = int(np.ceil(age_max))

    agegroups_inds_train = sort_subjectinds_oneyearagegroups(age_orig_train, age_min, age_max)

    agegroups_inds_outsample = sort_subjectinds_oneyearagegroups(age_orig_outsample, age_min, age_max)
    agegroups_inds_diag = []
    for i in range(n_diags):
        ag_here = sort_subjectinds_oneyearagegroups(age_orig_diag[i], age_min, age_max)
        agegroups_inds_diag.append(ag_here)

    n_agegroups = len(agegroups_inds_train)

    u_train_groups_sex0 = [[u_train[i,:] for i in range(n_subj) if (i in agegroups_inds_train[k] and sex_train[i]==0)] for k in range(n_agegroups)]
    u_train_groups_sex1 = [[u_train[i,:] for i in range(n_subj) if (i in agegroups_inds_train[k] and sex_train[i]==1)] for k in range(n_agegroups)]

    u_groupmeans_sex0 = np.zeros([n_agegroups, n_dir_used])
    u_groupmeans_sex1 = np.zeros([n_agegroups, n_dir_used])
    u_groupstds_sex0 = np.zeros([n_agegroups, n_dir_used])
    u_groupstds_sex1 = np.zeros([n_agegroups, n_dir_used])

    modelpred_train = fcn_obj.predict(X_dat_train)
    u_residuals_train = u_train[:,:n_dir_used] - modelpred_train[:,:n_dir_used]

    for k in range(n_agegroups):
        for j in range(n_dir_used):
            u_groupmeans_sex0[k,j] = np.mean([u_train_groups_sex0[k][i][j] for i in range(len(u_train_groups_sex0[k]))])
            u_groupmeans_sex1[k,j] = np.mean([u_train_groups_sex1[k][i][j] for i in range(len(u_train_groups_sex1[k]))])
            
    if not commonstd_io:
        for k in range(n_agegroups):
            for j in range(n_dir_used):
                u_groupstds_sex0[k,j] = np.std([u_train_groups_sex0[k][i][j] for i in range(len(u_train_groups_sex0[k]))], ddof=ddof_val)
                u_groupstds_sex1[k,j] = np.std([u_train_groups_sex1[k][i][j] for i in range(len(u_train_groups_sex1[k]))], ddof=ddof_val)
    else:
        commonstd = np.std(u_residuals_train,0, ddof=ddof_val)
        commonstd_mat = np.repeat(np.expand_dims(commonstd,0), n_agegroups, 0)
        u_groupstds_sex0[:,:] = commonstd_mat
        u_groupstds_sex1[:,:] = commonstd_mat

    x_agegroups = np.arange(int_min, int_max)+0.5
    x_agegroups_transf = standardize_age(x_agegroups, 1, dataset=dataset)
    x_agegroups_transf_sex0 = np.expand_dims(x_agegroups_transf,1)
    x_agegroups_transf_sex1 = np.expand_dims(x_agegroups_transf,1)

    if interaction_io:
        x_agegroups_transf_sex0 = np.concatenate((x_agegroups_transf_sex0, np.expand_dims(0*x_agegroups_transf,1)), axis=1)
        x_agegroups_transf_sex1 = np.concatenate((x_agegroups_transf_sex1, np.expand_dims(1*x_agegroups_transf,1)), axis=1)

    if age_squared_io:
        x_agegroups_transf_sex0 = np.concatenate((x_agegroups_transf_sex0, np.expand_dims(x_agegroups_transf**2,1)), axis=1)
        x_agegroups_transf_sex1 = np.concatenate((x_agegroups_transf_sex1, np.expand_dims(x_agegroups_transf**2,1)), axis=1)

    sex_covarind = covars_list.index('sex')
    X_dat_model_sex0 = np.insert(x_agegroups_transf_sex0, sex_covarind, np.zeros(len(x_agegroups)), axis=1)
    X_dat_model_sex1 = np.insert(x_agegroups_transf_sex1, sex_covarind, np.ones(len(x_agegroups)), axis=1)

    intercept_col = np.ones((X_dat_model_sex0.shape[0],1))
    X_dat_model_sex0 = np.concatenate([intercept_col, X_dat_model_sex0], axis=1)
    X_dat_model_sex1 = np.concatenate([intercept_col, X_dat_model_sex1], axis=1)

    if covars_list == ['sex', 'age'] or covars_list == ['age', 'sex']:
        y_pred_Z_sex0 = fcn_obj.predict(X_dat_model_sex0)
        y_pred_Z_sex1 = fcn_obj.predict(X_dat_model_sex1)
    else:
        X_dat_model_sex0 = float('nan')
        X_dat_model_sex1 = float('nan')
        y_pred_Z_sex0 = float('nan')
        y_pred_Z_sex1 = float('nan')


    Zscores_diag = [np.zeros([len(IDs_diags[j]), n_dir_used]) for j in range(n_diags)]

    modelpred_outsample = fcn_obj.predict(X_dat_outsample)

    # print(f'Test if outsample model prediction is the same with two methods: {np.all(modelpred_outsample==regmodel_preds_out)}')

    modelpred_diag = [fcn_obj.predict(X_dats_diag[j]) for j in range(n_diags)]
    # print(f'Test if diags model prediction is the same with two methods: {np.all(modelpred_diag==regmodel_pred_diags)}')


    u_residuals_outsample = u_outsample[:,:n_dir_used] - modelpred_outsample[j, :n_dir_used]
    u_residuals_diag = [u_diags[j][:,:n_dir_used] - modelpred_diag[j][:,:n_dir_used] for j in range(n_diags)]

    Zscores_outsample = np.zeros((len(IDs_outsample), n_dir_used))
    for i in range(n_outsample):
        sex_here = sex_outsample[i]
        age_group_here = [k for k in range(n_agegroups) if i in agegroups_inds_outsample[k]][0]
        if sex_here == 0:
            std_here = u_groupstds_sex0[age_group_here,:]
        elif sex_here ==1:
            std_here = u_groupstds_sex1[age_group_here,:]

        Zscores_outsample[i,:] = (u_outsample[i,:n_dir_used] - modelpred_outsample[i,:n_dir_used])/std_here

    for j in range(n_diags):
        for i in range(len(IDs_diags[j])):
            sex_here = sex_diag[j][i]
            age_group_here = [k for k in range(n_agegroups) if i in agegroups_inds_diag[j][k]][0]
            if sex_here == 0:
                std_here = u_groupstds_sex0[age_group_here,:]
            elif sex_here ==1:
                std_here = u_groupstds_sex1[age_group_here,:]

            Zscores_diag[j][i,:] = (u_diags[j][i,:n_dir_used] - modelpred_diag[j][i,:n_dir_used])/std_here


    var_names = ['IDs_list_train', 'IDs_outsample', 'IDs_diags', 'X_dat_train', 'X_dat_outsample', 'X_dats_diag', 'u_train', 'u_outsample', 'u_diag', 'age_orig_train', 'age_orig_outsample', 'age_orig_diag', 'X_dat_model_sex0', 'X_dat_model_sex1',  'R2_in', 'R2_out', 'R2s_diags', 'sex_train', 'sex_outsample', 'sex_diag', 'Zscores_outsample', 'Zscores_diag', 'diag_codes', 'dfd_vals_geom_train', 'x_agegroups', 'x_agegroups_transf', 'y_pred_Z_sex0', 'y_pred_Z_sex1', 'u_groupmeans_sex0', 'u_groupmeans_sex1', 'u_groupstds_sex0', 'u_groupstds_sex1', 'age_orig_train', 'age_orig_outsample', 'age_orig_diag', 'agegroups_inds_train', 'agegroups_inds_outsample', 'agegroups_inds_diag', 'u_residuals_train', 'modelpred_train', 'modelpred_outsample', 'modelpred_diag', 'MSE_in', 'MSE_out', 'MSEs_diags', 'u_residuals_outsample', 'u_residuals_diag', 'gamma_mat_train', 'beta_mat_train', 'set_seed', 'parc_list']
    variabs = [IDs_list_train, IDs_outsample, IDs_diags, X_dat_train, X_dat_outsample, X_dats_diag, u_train, u_outsample, u_diags, age_orig_train, age_orig_outsample, age_orig_diag, X_dat_model_sex0, X_dat_model_sex1,  R2_in, R2_out, R2s_diags, sex_train, sex_outsample, sex_diag, Zscores_outsample, Zscores_diag,  diag_codes, dfd_vals_geom_train, x_agegroups, x_agegroups_transf, y_pred_Z_sex0, y_pred_Z_sex1, u_groupmeans_sex0, u_groupmeans_sex1, u_groupstds_sex0, u_groupstds_sex1, age_orig_train, age_orig_outsample, age_orig_diag, agegroups_inds_train, agegroups_inds_outsample, agegroups_inds_diag, u_residuals_train, modelpred_train, modelpred_outsample, modelpred_diag, MSE_in, MSE_out, MSEs_diags, u_residuals_outsample, u_residuals_diag, gamma_mat_train, beta_mat_train, set_seed, parc_list]

    save_data_pickle(var_names=var_names, variabs=variabs, filename=savefilename)

    print(f'funcoin normative run took {time.time()-timeA}')


def funcoin_generate_filename(model_params, normative_io, longitudinal_io, filename_suf = '', model_type = '', n_dir_used=2, datpath = ''):

    n_init = model_params['n_init']
    n_subj = model_params['n_subj']
    sex = model_params['sex']
    site = model_params['site']
    n_parc = model_params['n_parc']
    n_dir = model_params['n_dir']
    covars_list = model_params['covars_list']
    interaction_io = model_params['interaction_io']
    age_squared_io = model_params['age_squared_io'] 
    dataset = model_params['dataset'] 
    brainhealthy_io = model_params['brainhealthy_io']
    betaLinReg = model_params['betaLinReg']
    system_name = platform.system()

    if datpath == '':
        if system_name == 'Linux':
            datpath = model_params['datpath_remote']
        elif system_name == 'Darwin':
            datpath = model_params['datpath_local']

    n_outsample = model_params['n_outsample']

    diag_codes = model_params['diag_codes']

    commonstd_io = model_params['commonstd_io']
    n_init = model_params['n_init']
    ts_stand_type = model_params['ts_stand_type']
    age_transf_type = model_params['age_transf_type']
    bootstrapCI_io = model_params['bootstrapCI_io']
    n_outsample_sets = model_params['n_outsample_sets']
    set_seed = model_params['set_seed_norm']


    if not model_type:
        filename = datpath + 'FUNCOIN_'
    else:
        filename = datpath + model_type + '_'

    filename += f'nsubj{n_subj}_ndir{n_dir}_sex{sex}_site{site}_nparc{n_parc}_ts{ts_stand_type}_agetrans{age_transf_type}'
    
    if not model_type:
        filename += f'_ninit{n_init}'

    filename += f'_bootCI{bootstrapCI_io}_covars'

    for i in range(len(covars_list)):
        filename = filename + covars_list[i]
    if interaction_io:
        filename = filename + '_interact'
    if age_squared_io:
        filename = filename + '_agesq'

    filename = filename + '_' + dataset

    if brainhealthy_io:
        filename += '_brainhealthy'

    n_diags = len(diag_codes)

    if normative_io:
        filename = filename + f'_normat_ndirused{n_dir_used}'
        
        if longitudinal_io:
            filename += '_LONGITUDINAL'
        else:
            filename += f'_nout{n_outsample}_noutsets{n_outsample_sets}_diags'
            for i in range(n_diags):
                if type(diag_codes[i]) == list:
                    filename = filename + 'c'
                    for i2 in range(len(diag_codes[i])):
                        filename += diag_codes[i][i2]
                    filename += 'c'
                else:
                    filename = filename + diag_codes[i]

            if commonstd_io:
                filename += '_comSTD'

    if betaLinReg:
        filename += '_betLR'

    filename += filename_suf

    return filename


def funcoin_runnew_splittest_par(model_params, test_mode = 0):
    """ Runs the splittest as described in the methods in the paper "One-shot normative modelling of whole-brain functional connectivity".
        Parameters:
        -----------
        model_params: Dictionary of model parameters. This can be loaded from the function get_model_parameters(model_version) found in current_model_parameters.py
        test_mode: Boolean or 0/1. If 1, only run with fewer subjects.

        Returns:
        --------
        Saves the model splittest results in a file in the folder specified by the string datpath. The following variables are saved: 
        'beta_mats_split_all1', 'beta_mats_split_all2', 'gamma_mats_split_all1', 'gamma_mats_split_all2', 'dfd_vals_geom_all1', 
        'dfd_vals_geom_all2', 'model_params', 'X_dats_all1', 'X_dats_all2'
    """

    if not test_mode:
        n_subj_list = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000]
    else:
        n_subj_list = [50, 100]


    n_init = model_params['n_init']
    rand_init = model_params['rand_init']
    covars_list = model_params['covars_list']
    # covars_list = ['age']
    ts_stand_type = model_params['ts_stand_type']
    age_transf_type = model_params['age_transf_type']
    seed_initial = model_params['seed_initial']
    datpath = model_params['datpath_remote']
    age_squared_io= model_params['age_squared_io']
    interaction_io = model_params['interaction_io']
    n_parc = model_params['n_parc']
    site = model_params['site']
    dataset = model_params['dataset']
    betaLinReg = model_params['betaLinReg']
    sex_choice = model_params['sex']

    datpath_temp = datpath + 'splittest/'

    set_seed_split_init = 185

    if not test_mode:
        n_tests = 100
        n_dir = 5
    else:
        n_tests = 5
        n_dir = 2


    for l in range(len(n_subj_list)):
        n_subj = n_subj_list[l]
        model_params['n_subj'] = n_subj
        n_per_split = n_subj//2

        set_seed_split_init + l*n_tests


        if n_per_split<16000:
            n_pool = 10
        else:
            n_pool = 4

        if sex_choice == False:
            covars_list = ['sex', 'age']
        else:
            covars_list = ['age']

        
        filename_suf = f'_splittest_ntest{n_tests}'

        filename = funcoin_generate_filename(model_params, 0, 0, filename_suf = filename_suf, n_dir_used=0, datpath=datpath_temp)
        filename_temp_init = funcoin_generate_filename(model_params, 0, 0, filename_suf = filename_suf, n_dir_used=0, datpath=datpath_temp)
                
    
        checkfile = os.path.isfile(filename)
        if not checkfile:  
            X_dat, Y_dat, IDs_list = prepare_run_funcoin(model_params, random_io = 0, seed_subj = 213+n_subj)

            filenames_temps = [filename_temp_init + str(k) for k in range(n_tests)]
            args_parallel = [(X_dat, Y_dat, n_dir, rand_init, n_init, seed_initial, set_seed_split_init, betaLinReg, dataset, filenames_temps[k]) for k in range(n_tests)]

            with Pool(processes=n_pool) as pool:
                pool.starmap(funcoin_splittest_singlerun_func, args_parallel)

            beta_mats_split_all1 = []
            beta_mats_split_all2 = []
            gamma_mats_split_all1 = []
            gamma_mats_split_all2 = []
            dfd_vals_geom_all1 = []
            dfd_vals_geom_all2 = []
            X_dats_all1 = []
            X_dats_all2 = []
            for i in range(n_tests):
                filename_par = filename_temp_init + str(i)
                
                var_names, var_dic = load_data_pickle(filename_par)
                beta_mat1 = var_dic['beta_mat1']
                beta_mat2 = var_dic['beta_mat2']
                gamma_mat1 = var_dic['gamma_mat1']
                gamma_mat2 = var_dic['gamma_mat2']
                dfd_vals_geom1 = var_dic['dfd_vals_geom1']
                dfd_vals_geom2 = var_dic['dfd_vals_geom2']
                X_dat1 = var_dic['X_dat1']
                X_dat2 = var_dic['X_dat2']

                beta_mats_split_all1.append(beta_mat1)
                beta_mats_split_all2.append(beta_mat2)
                gamma_mats_split_all1.append(gamma_mat1)
                gamma_mats_split_all2.append(gamma_mat2)
                dfd_vals_geom_all1.append(dfd_vals_geom1)
                dfd_vals_geom_all2.append(dfd_vals_geom2)
                X_dats_all1.append(X_dat1)
                X_dats_all2.append(X_dat2)

                os.remove(filename_par)

            variabs = [beta_mats_split_all1, beta_mats_split_all2, gamma_mats_split_all1, gamma_mats_split_all2, dfd_vals_geom_all1, dfd_vals_geom_all2, model_params, X_dats_all1, X_dats_all2]
            var_names = ['beta_mats_split_all1', 'beta_mats_split_all2', 'gamma_mats_split_all1', 'gamma_mats_split_all2', 'dfd_vals_geom_all1', 'dfd_vals_geom_all2', 'model_params', 'X_dats_all1', 'X_dats_all2']
            
            save_data_pickle(var_names, variabs, filename)




#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
