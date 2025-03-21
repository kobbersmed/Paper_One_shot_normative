import numpy as np
from io_funcs import load_data_pickle, save_data_pickle, load_data_stand_calc_FC
from funcoin_runscripts import funcoin_generate_filename
from FC_funcs import corr_to_z
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

def run_LinReg_normative(model_params, test_mode = 0, corr_to_z_io = 1, filename_suf = ''):
    """ Runs the FUNCOIN decomposition with the specified parameters and saves the data in the folder specified by datpath.
        Parameters:
        -----------
        model_params: Dictionary of model parameters. This can be loaded from the function get_model_parameters(model_version) found in current_model_parameters.py
        test_mode: Boolean or 0/1. If 1, only run with fewer subjects.
        corr_to_z_io: Boolean or 0/1. Specifies whether to use the Fisher transformation on the FC edges before fitting the model.
        filename_suf: String 

        Returns:
        --------
        Saves the linear regression model fitting and analysis results in a file in the folder specified by the string datpath. The following variables are saved: 
        X_dat_train, X_dats_outsample, X_dats_diags, SDs_train, residuals_train, residuals_outsample, residuals_diags, prediction_train, 
        predictions_outsample, prediction_diags, Zscores_train, Zscores_outsample, Zscores_diags, coefs, MSE_edgewise_train, R2_edgewise_train,
        MSE_edgewise_sexes_train, R2_edgewise_sexes_train, MSE_edgewise_outsample, MSE_edgewise_outsample_sexes, R2_edgewise_outsample, R2_edgewise_outsample_sexes, 
        MSE_overall_train, MSE_overall_outsample, R2_overall_train, R2_overall_outsample, sex_train, sex_outsample, sex_diag, Y_train, Y_outsample, Y_diags

    """

    modeltype = 'LR'

    n_outsample_sets = model_params['n_outsample_sets']
    diag_labels = model_params['diag_labels']

    n_dirs_used = 2

    filename_norm = funcoin_generate_filename(model_params, 1, 0, filename_suf = filename_suf, model_type = '', n_dir_used=n_dirs_used)

    if test_mode:
        filename_suf = '_TESTMODE'
    else:
        filename_suf = ''

    savefilename = funcoin_generate_filename(model_params, 1, 0, filename_suf = filename_suf, model_type = modeltype, n_dir_used=n_dirs_used)

    if corr_to_z_io:
        savefilename += '_RtoZ'

    check1 = os.path.isfile(savefilename)



    if check1:
        print(f'LinReg/PCR result file found.')
        print('Skipping script')
        return []


    var_names_norm, var_dic_norm = load_data_pickle(filename_norm)

    #Load in variables from result files



    IDs_train = var_dic_norm['IDs_list_train']
    IDs_outsample = var_dic_norm['IDs_outsample']
    IDs_diags = var_dic_norm['IDs_diags']

    X_dat_train = var_dic_norm['X_dat_train']
    X_dat_outsample = var_dic_norm['X_dat_outsample']
    X_dats_diags = var_dic_norm['X_dats_diag']
    sex_train = var_dic_norm['sex_train']
    sex_outsample = var_dic_norm['sex_outsample']
    sex_diag = var_dic_norm['sex_diag']

    if test_mode:
        n_outsample = 20
        n_outsample_sets = 2
        X_dat_train = X_dat_train[:100,:]
        X_dats_outsample = X_dats_outsample[:n_outsample,:]
        IDs_train = IDs_train[:100]
        IDs_outsample = IDs_outsample[:n_outsample]
        sex_train = sex_train[:100]
        sex_outsample = sex_outsample[:n_outsample]
        sex_diag = sex_diag[:2]
        IDs_diags = IDs_diags[:2]
        X_dats_diags = X_dats_diags[:2]

    del var_names_norm, var_dic_norm

    FC_mat_all_train = load_data_stand_calc_FC(data_type = 'training', IDs=IDs_train, FC_type = 'Pearson')

    FC_mats_all_outsample = load_data_stand_calc_FC(data_type = 'testing', IDs=IDs_outsample, FC_type = 'Pearson')

    FC_mat_all_diags = [load_data_stand_calc_FC(data_type = diag_labels[i], IDs=IDs_diags[i], FC_type = 'Pearson') for i in range(len(IDs_diags))]

    if corr_to_z_io:
        Y_train = corr_to_z(FC_mat_all_train)
        Y_outsample = [corr_to_z(FC_mats_all_outsample[i]) for i in range(len(FC_mats_all_outsample))]
        Y_diags = [corr_to_z(FC_mat_all_diags[i]) for i in range(len(FC_mat_all_diags))]
    else:
        Y_train = FC_mat_all_train
        Y_outsample = FC_mats_all_outsample
        Y_diags = FC_mat_all_diags


    #Train model, prediction, residuals, scores
    regmodel = LinearRegression().fit(X_dat_train, Y_train)

    print(X_dat_train.shape)

    prediction_train = regmodel.predict(X_dat_train)
    residuals_train = Y_train - prediction_train

    predictions_outsample = regmodel.predict(X_dats_outsample)
    residuals_outsample = Y_outsample - predictions_outsample

    prediction_diags = [regmodel.predict(X_dats_diags[i]) for i in range(len(IDs_diags))]
    residuals_diags = [Y_diags[i] - prediction_diags[i] for i in range(len(IDs_diags))]



    MSE_edgewise_train = np.array([mean_squared_error(Y_train[:,i], prediction_train[:,i]) for i in range(Y_train.shape[1])])
    R2_edgewise_train = np.array([r2_score(Y_train[:,i], prediction_train[:,i]) for i in range(Y_train.shape[1])])

    MSE_edgewise_sexes_train = np.array([[mean_squared_error(Y_train[sex_train==k,i], prediction_train[sex_train==k,i]) for i in range(Y_train.shape[1])] for k in [0,1]])
    R2_edgewise_sexes_train = np.array([[r2_score(Y_train[sex_train==k,i], prediction_train[sex_train==k,i]) for i in range(Y_train.shape[1])] for k in [0,1]])

    R2_edgewise_outsample = np.array([r2_score(Y_outsample[:,i], predictions_outsample[:,i]) for i in range(Y_outsample.shape[1])])
    R2_edgewise_outsample_sexes = np.array([[r2_score(Y_outsample[sex_outsample==k, i], predictions_outsample[sex_outsample==k, i]) for i in range(Y_outsample.shape[1])] for k in [0,1]])

    MSE_edgewise_outsample = np.array([mean_squared_error(Y_outsample[:,i], predictions_outsample[:,i]) for i in range(Y_outsample.shape[1])])
    MSE_edgewise_outsample_sexes = np.array([[mean_squared_error(Y_outsample[sex_outsample==k, i], predictions_outsample[sex_outsample==k, i]) for i in range(Y_outsample.shape[1])] for k in [0,1]])


    MSE_overall_train = mean_squared_error(Y_train, prediction_train)
    MSE_overall_outsample = mean_squared_error(Y_outsample, predictions_outsample)
    R2_overall_train = r2_score(Y_train, prediction_train)
    R2_overall_outsample = r2_score(Y_outsample, predictions_outsample)


    # #Determine edges with highest slopes and highest scores  

    coefs = regmodel.coef_.copy()


    #SD and Z-scores for training, outsample and diags
    SDs_train = np.std(residuals_train, axis = 0, ddof=1)

    Zscores_train = np.array([residuals_train[i]/SDs_train for i in range(Y_train.shape[0])])
    Zscores_outsample = np.array([residuals_outsample[i]/SDs_train for i in range(Y_outsample.shape[0])]) 
    Zscores_diags = [np.array([residuals_diags[k][i]/SDs_train for i in range(Y_diags[k].shape[0])]) for k in range(len(IDs_diags))]


    variabs_save = [X_dat_train, X_dats_outsample, X_dats_diags, SDs_train, residuals_train, residuals_outsample, residuals_diags, prediction_train, predictions_outsample, prediction_diags, Zscores_train, Zscores_outsample, Zscores_diags, coefs, MSE_edgewise_train, R2_edgewise_train, MSE_edgewise_sexes_train, R2_edgewise_sexes_train, MSE_edgewise_outsample, MSE_edgewise_outsample_sexes, R2_edgewise_outsample, R2_edgewise_outsample_sexes, MSE_overall_train, MSE_overall_outsample, R2_overall_train, R2_overall_outsample, sex_train, sex_outsample, sex_diag, Y_train, Y_outsample, Y_diags]
    var_names_save = ['X_dat_train', 'X_dats_outsample', 'X_dats_diags', 'SDs_train', 'residuals_train', 'residuals_outsample', 'residuals_diags', 'prediction_train', 'predictions_outsample', 'prediction_diags', 'Zscores_train', 'Zscores_outsample', 'Zscores_diags', 'coefs', 'MSE_edgewise_train', 'R2_edgewise_train', 'MSE_edgewise_sexes_train', 'R2_edgewise_sexes_train', 'MSE_edgewise_outsample', 'MSE_edgewise_outsample_sexes', 'R2_edgewise_outsample', 'R2_edgewise_outsample_sexes', 'MSE_overall_train', 'MSE_overall_outsample', 'R2_overall_train', 'R2_overall_outsample', 'sex_train', 'sex_outsample', 'sex_diag', 'Y_train', 'Y_outsample', 'Y_diags']

    save_data_pickle(var_names_save, variabs_save, savefilename)

