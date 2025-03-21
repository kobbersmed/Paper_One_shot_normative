
import numpy as np

def get_model_parameters(model_version):
    
    ###In this function most parameters of the Funcoin normative model are defined.   

    ###MODEL VERSIONS
    #0: Standard model
    #1: Model with site covariate in addition to sex and age
    #2: Standard model with bootstrapping confidence intervals of beta coefficients

  
    model_vars = {}


    ###Fixed, "basic" parameters
    model_vars['n_init'] = 20  # Try 20 different initial conditions
    model_vars['ts_stand_type'] = 2  # Standardize the time series data to mean 0 var 1
    model_vars['age_transf_type'] = 1  # Linear transformation of age variable to the interval [0,1]
    model_vars['rand_init'] = 1  # Random initial conditions 
    model_vars['n_bootstrap'] = 500  # Number of bootstrap samples. Only used if bootstrapCI_io == 1
    model_vars['CI_lvl'] = 0.05  # Confidence interval level. Only used if bootstrapCI_io == 1. 

    ###Parameters for funcoin normative model 
    model_vars['n_subj'] = 32000  # Number of subjects, training set
    model_vars['sex'] = False  # Both sexes included in subject dataset 
    model_vars['site'] = False  # Subjects are included regardless of scanning site
    model_vars['n_parc'] = 100  # UKB ICA-parcellation 100 (55 components after artifact removal)
    model_vars['n_dir'] = 5  # Maximal number of directions identified
    model_vars['interaction_io'] = 1  # Sex*age interaction
    model_vars['age_squared_io'] = 0  # Not including squared age
    model_vars['betaLinReg'] = 1  # Using the funcoin variant of the algorithm, i.e. the unbiased version
    model_vars['set_seed_norm'] = 100  # Seed for picking test subjects 
    model_vars['seed_initial'] = 12345  # Seed of initial conditions
    model_vars['FC_type'] = 'Pearson' 
    model_vars['n_outsample'] = 14000  # No of test subjects
    model_vars['n_outsample_sets'] = 1  # All test subjects pooled together
    diag_codes = ['F31', 'G20', 'G35']  # Diagnoses used in manuscript
    model_vars['diag_codes'] = diag_codes
    diag_labels = ['BP', 'PD', 'MS']
    model_vars['diag_labels'] = diag_labels
    model_vars['commonstd_io'] = 1  # Assume variance homogeneity of transformed values
    if model_version ==1:
        model_vars['covars_list'] = ['sex', 'age', 'site']
    else:
        model_vars['covars_list'] = ['sex', 'age']

    if model_version == 2:
        model_vars['bootstrapCI_io'] = 1 
    else:
        model_vars['bootstrapCI_io'] = 0

    model_vars['dataset'] = 'UKB'
    model_vars['brainhealthy_io'] = 1  # Only including training subjects with no brain diagnoses (ICD-10 F or G) 

    #Define path to save data
    if (model_version == 0) or (model_version == 2):
        model_vars['datpath_remote'] = './paper_data/'
        model_vars['datpath_local'] = './paper_data/'
    elif model_version == 1:
        model_vars['datpath_remote'] = './paper_data_site/'
        model_vars['datpath_local'] = './paper_data_site/'
      
            

    
    return model_vars

