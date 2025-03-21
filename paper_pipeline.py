from current_model_parameters import get_model_parameters
from funcoin_runscripts import run_funcoin_method, funcoin_run_normative_UKB_diag, funcoin_runnew_splittest_par
from LinReg_normative import run_LinReg_normative
from io_funcs import get_new_IDs_healthy
from generate_IDfile import gen_IDs_training

###MODEL VERSIONS
#0: Standard model
#1: Model with site covariate in addition to sex and age
#2: Standard model with bootstrapping confidence intervals of beta coefficients

model_versions = [0,1]

for model_version_choice in model_versions:

    
    model_params = get_model_parameters(model_version_choice)

    ####Basic parameters for FUNCOIN model

    n_subj = model_params['n_subj']
    sex = model_params['sex'] #False, because we are sampling subjects regardless of sex
    site = model_params['site'] #False, because we are sampling subjects regardless of scanning site
    dataset = model_params['dataset'] #Specifies the dataset to be used
    brainhealthy_io = model_params['brainhealthy_io'] #Specifies that subjects are considered healthy if they have no diagnoses in category F and G (as opposed to considering only subjects without any diagnosis)
    datpath = model_params['datpath_remote']


    ##Load reproducible list of IDs for training
    IDs_list_train, _ = gen_IDs_training(n_subj, datpath, set_seed=1234, dataset=dataset)

    # IDs_list_train = get_new_IDs_healthy(n_subj=n_subj, IDs_old=[], sex=sex, site=site, brainhealth_only=brainhealthy_io, random_io = False, set_seed = 1234, dataset = dataset)

    filename_suf = '_TESTINGPAPERSCRIPTS' #Specify this string to append it to the end of the filename

    print('Checking result file. If no file, then running decomposition')
    run_funcoin_method(model_params, IDs_list = IDs_list_train, parc_list = [], filename_suf = filename_suf) #Runs the funcoin decomposition
    print('Done.')
    n_dir_used = 2
    print(f'Checking result file. If no file, then running normative. n_dir_used = {n_dir_used}')
    print('Only main diagnoses')
    funcoin_run_normative_UKB_diag(model_params, n_dir_used = n_dir_used, IDs_used = IDs_list_train, parc_list = [], filename_suf = filename_suf) #Loads the result file from funcoin decomposition and calculates various analyses for the normative modelling
    print('DONE')
    
    ####

    # #LinReg
    n_dir_used = 2
    print(f'Checking result file. If no file, then running LR.')
    print('Only main diagnoses')
    run_LinReg_normative(model_params, test_mode = 0, corr_to_z_io=1, filename_suf=filename_suf) #An edgewise linear regression-based normative model after Fisher's r-to-z transformation. 
    print('DONE')


    ###Run FUNCOIN decomp again without betaLR for making biasgraphs (Figure S1B)
    model_params_alt = model_params.copy()
    model_params_alt['betaLinReg'] = 0

    run_funcoin_method(model_params_alt, IDs_list = IDs_list_train, parc_list = [], filename_suf = filename_suf + '_COEFS_NOLINREG') #Running the funcoin decomposition without the bias correction. Results used in Fig S1B

    if model_version_choice == 0:
        ###Run split test to asses consistency
        print('Checking result file. If no file, then running splittest for assesing consistency.')
        funcoin_runnew_splittest_par(model_params, test_mode=1) #Runs splittest for assesment of consistency of the method. Visualized in Fig S1A
        print('Splittest DONE')