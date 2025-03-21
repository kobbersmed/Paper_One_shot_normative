import numpy as np
import warnings
from FC_funcs import calc_FCmatrix_all_fromlist
# import h5py
import pickle
import platform
import csv
from util_ts_functions import truncate_ts_all, standardize_ts

###Functions for preparing running FUNCOIN scripts

def prepare_run_funcoin(model_params, parc_list = [], random_io=False, seed_subj = None):

    n_subj = model_params['n_subj']
    covars_list = model_params['covars_list']
    # covars_list = ['age']
    ts_stand_type = model_params['ts_stand_type']
    age_transf_type = model_params['age_transf_type']
    age_squared_io= model_params['age_squared_io']
    interaction_io = model_params['interaction_io']
    n_parc = model_params['n_parc']
    site = model_params['site']
    dataset = model_params['dataset']
    sex = model_params['sex']
    brainhealthy_io = model_params['brainhealthy_io'] 


    if brainhealthy_io:
        IDs_list = get_new_IDs_healthy(n_subj, IDs_old=[], sex=sex, site=site, brainhealth_only=True, random_io = random_io, set_seed = seed_subj, dataset = 'UKB')
    else:
        if (type(sex) == int) and (not site):
            IDs_list = get_IDs_sex(n_subj, sex, dataset=dataset, random_io=random_io, set_seed=seed_subj)
        elif (sex == False) and (not site):
            IDs_list = get_IDs(n_subj, dataset=dataset, random_io=random_io, set_seed=seed_subj)
        elif site:
            IDs_list = get_IDs_stratify(n_subj, sex = sex, site = site, dataset=dataset, random_io=random_io, set_seed=seed_subj)

    Y_dat = load_data_stand_list(IDs_list, no_parc= n_parc, ts_stand_type=ts_stand_type, remove_startinds = 1, remove_badparc = 1, dataset=dataset, subparc=parc_list)

    # data_coll_stand = load_dr_data_standard(IDs_list, no_parc = 25)

    # FC_matrix_all_list = calc_FCmatrix_all_list(IDs_list, data_coll_stand, FC_type)
    # cov_sample_ave_all = calc_ref_matrix(FC_matrix_all_list, ref_type='euclidian')

    X_dat = prepare_X_regres_covars(IDs_list, covars_list, dataset=dataset, age_transf_type=age_transf_type)

    if interaction_io:
        age_ind = covars_list.index('age') +1
        sex_ind = covars_list.index('sex') +1
        X_dat = np.concatenate([X_dat, np.expand_dims(X_dat[:,age_ind]*X_dat[:,sex_ind],1)],1)

    if age_squared_io:
        age_ind = covars_list.index('age') +1
        X_dat = np.concatenate([X_dat, np.expand_dims(X_dat[:,age_ind]**2,1)],1)

    return X_dat, Y_dat, IDs_list

def funcoin_prepare_from_IDslist(IDs_list, covars_list, n_parc, ts_stand_type = 2, age_transf_type = 1, interaction_io=0, age_squared_io = 0, dataset='UKB', parc_list = []):

    Y_dat = load_data_stand_list(IDs_list, no_parc= n_parc, ts_stand_type=ts_stand_type, remove_startinds = 1, remove_badparc = 1, dataset = dataset, subparc=parc_list)

    # data_coll_stand = load_dr_data_standard(IDs_list, no_parc = 25)

    # FC_matrix_all_list = calc_FCmatrix_all_list(IDs_list, data_coll_stand, FC_type)
    # cov_sample_ave_all = calc_ref_matrix(FC_matrix_all_list, ref_type='euclidian')

    X_dat = prepare_X_regres_covars(IDs_list, covars_list, dataset=dataset, age_transf_type=age_transf_type)

    if interaction_io:
        age_ind = covars_list.index('age') +1
        sex_ind = covars_list.index('sex') +1
        X_dat = np.concatenate([X_dat, np.expand_dims(X_dat[:,age_ind]*X_dat[:,sex_ind],1)],1)

    if age_squared_io:
        age_ind = covars_list.index('age') +1
        X_dat = np.concatenate([X_dat, np.expand_dims(X_dat[:,age_ind]**2,1)],1)

    return X_dat, Y_dat

def prepare_X_regres_covars(IDs_list, covars_list, dataset, age_transf_type = 1):

    covar_vals = read_behavioural_csv(IDs_list, covars_list, dataset = dataset)
    if 'age' in covars_list:
        age_ind = covars_list.index('age')
        age_var = covar_vals[:,age_ind]
        age_stand = standardize_age(age_var, transf_type=age_transf_type, dataset=dataset)
        covar_vals[:,age_ind] = age_stand
        
    if 'site' in covars_list:
        site_ind = covars_list.index('site')
        sites_val = covar_vals[:,site_ind]
        covar_vals = np.delete(covar_vals, site_ind, 1)
        n_sites = round(np.maximum(np.max(sites_val),4))
        n_subj = covar_vals.shape[0]
        site_covar = np.zeros([n_subj, n_sites-1])
        for i in range(1,n_sites):
            site_var = np.zeros(n_subj)
            site_var = np.where(sites_val == i+1, 1, 0)
            site_covar[:,i-1] = site_var
        covar_vals = np.concatenate([covar_vals, site_covar],1)


    X_dat = np.concatenate([np.expand_dims(np.ones(covar_vals.shape[0]),1), covar_vals],1)

    return X_dat


###ID list functions:

def get_IDs(n_subj = float('inf'), ignore_warning = False, dataset = 'UKB', firstscanonly = True, random_io = False, set_seed = None):
    
    #The bad IDs are IDs for which we don't have extended behaviourals like diagnoses etc.
    bad_IDs = ['1697911', '1202715', '2377684', '2135288', '2623882', '3622766', '4612030', '4155777', '4070129', '5293448', '5380068']
    #Also exclude IDs with fMRI time series but without basic behavioural data
    bad_IDs.extend(['3150585', '3335139', '4065532', '4909374'])


    system_name = platform.system()

    IDsfilename = '/media/raw_data/UKB_2/ANALYSIS/IDs_list_fMRI.txt'

    with open(IDsfilename) as fID:
        IDs_list = fID.read().splitlines()
    
    for i in range(len(bad_IDs)): #Remove IDs for which we don't have the newest behaviourals, e.g. diagnoses
        badID1 = '2' + bad_IDs[i]
        if badID1 in IDs_list:
            IDs_list.remove(badID1)
        badID2 = '3' + bad_IDs[i]
        if badID2 in IDs_list:
            IDs_list.remove(badID2)

    if firstscanonly:
        IDs_list = [IDs_list[i] for i in range(len(IDs_list)) if IDs_list[i][0] == '2']


    n_total = min(n_subj, len(IDs_list))

    if random_io == True:
        rng = np.random.default_rng(seed = set_seed)
        IDs_inds = rng.choice(len(IDs_list), size = n_total, replace=False)
    else:
        IDs_inds = range(n_total)

    IDs_list = [IDs_list[i] for i in IDs_inds]
    
    if (len(IDs_list) < n_subj) and (n_subj < float('inf')) and (ignore_warning == False):
        print('Number of IDs found was smaller than specified n_subj.')
        print(f'Returned a list of {len(IDs_list)} IDs')

    return IDs_list

def get_twoscan_IDs(n_subj= float('inf'), random_io = False, set_seed = None, dataset = 'UKB'):
    all_IDs = get_IDs(firstscanonly = False, random_io=random_io, set_seed = set_seed,dataset=dataset)
    IDs_3 = [all_IDs[i] for i in range(len(all_IDs)) if all_IDs[i][0] == '3']
    IDs_2_3_intersect = [IDs_3[i][1:] for i in range(len(IDs_3)) if ('2' +IDs_3[i][1:]) in all_IDs]

    IDs_list = [['2' + IDs_2_3_intersect[i] for i in range(len(IDs_2_3_intersect))], ['3' + IDs_2_3_intersect[i] for i in range(len(IDs_2_3_intersect))]]
    
    if len(IDs_list[1]) > n_subj:
        IDs_two_scan = [IDs_list[0][:n_subj], IDs_list[0][:n_subj]]
    elif (len(IDs_list[1]) < n_subj) and (n_subj < float('inf')):
        IDs_two_scan = IDs_list
        print('Number of IDs found was smaller than specified n_subj.')
        print(f'Returned a list of {len(IDs_list)} IDs')
    else:
        IDs_two_scan = IDs_list
        
    return IDs_two_scan

def get_new_IDs(n_subj, IDs_list_old, firstscanonly = True, sex = False, site = False, dataset = 'UKB', random_io = False, set_seed = None):
    """
    Returns a list of IDs which are not already on the inputted list of IDs.
    """

    if not sex and not site:
        IDs_list_all = get_IDs(float('inf'), firstscanonly = firstscanonly, dataset = dataset, random_io=random_io, set_seed = set_seed)
    elif sex and not site:
        IDs_list_all = get_IDs_sex(float('inf'), sex = sex, firstscanonly = firstscanonly, dataset = dataset, random_io=random_io, set_seed = set_seed)
    elif site:
        IDs_list_all = get_IDs_stratify(float('inf', sex = sex, site = site), dataset = dataset, firstscanonly = firstscanonly, random_io=random_io, set_seed = set_seed)

    IDs_list_init = [IDs_list_all[i] for i in range(len(IDs_list_all)) if IDs_list_all[i] not in IDs_list_old ]
    
    if len(IDs_list_init) >= n_subj:
        IDs_list_new = IDs_list_init[:n_subj]
    else:
        IDs_list_new = IDs_list_init
        warnings.warn(f'Requested more IDs than was available. Returned a list of length {len(IDs_list_new)}.')

    return IDs_list_new

def get_IDs_sex(n_subj=float('inf'), sex=0, firstscanonly = True, random_io = False, dataset = 'UKB', set_seed=None):
    #Gets a list of the first n_subj subject IDs of specified sex
    # 
    ctrl = 0

    IDs_list_init = get_IDs(float('inf'),  firstscanonly = firstscanonly, dataset=dataset, random_io=random_io, set_seed=set_seed)
    sex_var = read_behavioural_csv(IDs_list_init, covars = ['sex'], dataset=dataset)
    sex_inds = [i for i in range(len(sex_var)) if sex_var[i] == sex]
    IDs_list_sex = [IDs_list_init[i] for i in sex_inds]

    if len(sex_inds)<n_subj:
        warnings.warn(f'Requested more IDs than possible. Returned {len(sex_inds)} subject IDs.')
        
    if n_subj<len(IDs_list_sex):
        IDs_sex = IDs_list_sex[:n_subj]
    else:
        IDs_sex = IDs_list_sex

    return IDs_sex

def get_IDs_stratify(n_subj=float('inf'), age_int = False, sex = False, site = False, firstscanonly = True, dataset = 'UKB', random_io = False, set_seed=None):
    
    if dataset == 'HCP':
        raise Exception('The function get_IDs_stratify only works for UKB at the moment')

    pos_args = ['age', 'sex', 'site']
    posarg_vals = [age_int, sex, site]

    used_args = [pos_args[i] for i in range(len(pos_args)) if posarg_vals[i] is not False]
    no_args = len(used_args)
    ctrl = 0

    if n_subj != float('inf'):
        n_total = round((no_args+1)*n_subj)
    else:
        n_total = n_subj

    while ctrl==0:
        IDs_list = get_IDs(n_total, ignore_warning=True, firstscanonly = firstscanonly, random_io=random_io, set_seed = set_seed)
        all_inds = [np.arange(len(IDs_list))]
        covar_vals = read_behavioural_csv(IDs_list, covars = used_args)
        if sex is not False:
            sex_arg_ind = used_args.index('sex')
            sex_var = covar_vals[:,sex_arg_ind]
            sex_inds = [i for i in range(len(sex_var)) if sex_var[i] == sex]
            all_inds.append(sex_inds)

        if site is not False:
            site_arg_ind = used_args.index('site')
            site_var = covar_vals[:, site_arg_ind]
            site_inds = [i for i in range(len(site_var)) if site_var[i] == site]
            all_inds.append(site_inds)

        if age_int is not False:
            age_arg_ind = used_args.index('age')
            age_var = covar_vals[:,age_arg_ind]
            age_inds = [i for i in range(len(age_var)) if (age_var[i]>=age_int[0] and age_var[i] < age_int[1])]
            all_inds.append(age_inds)


        
        common_inds_set = set(all_inds[0])
        for i in range(no_args):
            common_inds_set = set(common_inds_set).intersection(set(all_inds[i+1]))
        common_inds = list(common_inds_set)
        common_inds.sort()


        if len(common_inds)<n_subj and n_total==len(IDs_list):
            n_total = round(n_total * 1.5)
        else:
            if len(common_inds)<n_subj:
                warnings.warn(f'Requested more IDs than possible. Returned {len(common_inds)} subject IDs.')
            
            IDs_strati = [IDs_list[i] for i in common_inds]
            min_ind = round(np.minimum(n_subj, len(common_inds)))
            IDs_strat = IDs_strati[:min_ind]
            ctrl = 1
            
    return IDs_strat

def get_new_IDs_healthy(n_subj=float('inf'), IDs_old=[], sex=False, site=False, brainhealth_only=True, random_io = False, set_seed = None, dataset = 'UKB'):
    #Dataset is UKB
    var_names = get_varnames_UKB(var_type = 'nonim')
    diagnoses_inds = []

    if brainhealth_only:
        for i in range(len(var_names)):
            test1 = var_names[i].find('Diagnoses - ICD10 (F')
            test2 = var_names[i].find('Diagnoses - ICD10 (G')

            if test1>=0 or test2>=0:
                diagnoses_inds.append(i)
    else:
        for i in range(len(var_names)):
            test1 = var_names[i].find('Diagnoses - ICD10 (')

            if test1>=0:
                diagnoses_inds.append(i)

    diagnoses_varnames = [var_names[i] for i in diagnoses_inds]

    if (sex is False) and (site == False):
        IDs_all = get_IDs(dataset=dataset, random_io=random_io, set_seed=set_seed)
    elif site == False:
        IDs_all = get_IDs_sex(sex=sex, dataset=dataset, random_io=random_io, set_seed=set_seed)
    else:
        IDs_all = get_IDs_stratify(sex=sex, site=site, dataset=dataset, random_io=random_io, set_seed=set_seed)

    IDs_new = [IDs_all[i] for i in range(len(IDs_all)) if IDs_all[i] not in IDs_old]

    # diagnose_vars = read_othervars_UKB(IDs_new, varnames = diagnoses_varnames, var_type='nonim')
    
    # IDs_healthy = [IDs_new[i] for i in range(len(IDs_new)) if not np.any(diagnose_vars[i,:])]
    
    varsfilename = '/media/raw_data/UKB_2/ANALYSIS/non_imaging_UKB2_Janus.csv'

    no_subj = len(IDs_new)
    IDs_new_mod = [IDs_new[i][1:] for i in range(no_subj)] #Removing session-number prefix to obtain raw ID

    with open(varsfilename, 'r', encoding='UTF8') as covarcsv:
        csvreader = csv.reader(covarcsv)
        ctrl = 0
        for row in csvreader:
            if ctrl == 0:
                covar_names = row
                covar_inds = [covar_names.index(diagnoses_varnames[i]) for i in range(len(diagnoses_varnames))]

                ctrl = 1
            else:
                if str(row[0]) in IDs_new_mod:
                    row_vals = np.array([float(row[i]) for i in covar_inds])
                    if np.any(row_vals):
                        ID_ind = IDs_new_mod.index(str(row[0]))
                        IDs_new[ID_ind] = False
    
    IDs_healthy = [IDs_new[i] for i in range(len(IDs_new)) if IDs_new[i]!=False]


    if n_subj<float('inf') and len(IDs_healthy)>n_subj:
        IDs_healthy_new = IDs_healthy[:n_subj]
    else:
        print(f'Requested more IDs than possible. Returned a list of {len(IDs_healthy)}')
        IDs_healthy_new = IDs_healthy

    return IDs_healthy_new



####Behavioural data

def read_behavioural_csv(IDs_list = ['21823493', '21380107'], covars=['age', 'sex'], dataset = 'UKB', covars_version = 'new'):

    system_name = platform.system()


    if covars_version == 'old':
        covarfilename = '/media/raw_data/UK_Biobank/Behavioural/behavioural_UBK_Janus.csv'
    elif covars_version == 'new':
        covarfilename = '/media/raw_data/UKB_2/ANALYSIS/basicbehaviourals_UKB2_Janus.csv'


    # if covars_version == 'new':
    #     IDs_list_mod = [IDs_list[i][1:] for i in range(len(IDs_list)) if len(IDs_list[i])==8]
    # else:
    IDs_list_mod = IDs_list

    no_subj = len(IDs_list)
    variabs_temp = []

    with open(covarfilename, 'r', encoding='UTF8') as covarcsv:
        csvreader = csv.reader(covarcsv)
        ctrl = 0
        for row in csvreader:
            if ctrl == 0:
                covar_names = row
                covar_inds = [covar_names.index(covars[i]) for i in range(len(covars))]

                ctrl = 1
            else:
                if str(row[0]) in IDs_list_mod:
                    if (dataset == 'UKB') or (dataset == 'UKB_surf'):
                        row_vals = [int(row[0]), float(row[1]), int(row[2]), int(row[3])]
                    elif dataset == 'HCP':
                        row_vals = row
                    variabs_temp.append(row_vals)

    if dataset == 'HCP':
        covars_list = []
        covar_float = [[float(variabs_temp[k][i]) for i in range(len(variabs_temp[0]))] for k in range(len(variabs_temp))]

        for k in range(len(variabs_temp)):
            rowval = []
            for i in range(len(covar_float[k])):
                try:
                    test_val = int(covar_float[k][i])
                except:
                    rowval.append(covar_float[k][i])
                else:
                    if test_val == covar_float[k][i]:
                        rowval.append(test_val)
                    else:
                        rowval.append(covar_float[k][i])
            covars_list.append(rowval)
        variabs_temp = covar_float

    variabs = np.array(variabs_temp, dtype=object)


    covar_vals = np.zeros([no_subj, len(covars)])
    IDs_array = np.array(IDs_list, dtype = int)
    variabs_IDs_list = [str(variabs[i,0]) for i in range(variabs.shape[0])]
    for i in range(no_subj):
        subj_ind = variabs_IDs_list.index(IDs_list[i]) #np.where(variabs[:,0] == IDs_array[i])[0]
        for k in range(len(covars)):
            covar_nam = covars[k]
            covar_ind = covar_names.index(covar_nam)
            covar_vals[i,k] = variabs[subj_ind, covar_ind]
    
    return covar_vals

def read_othervars_UKB(IDs_list, varnames, var_type='nonim'):
    #var_type is 'nonim' (non-imaging) or 'Svars'

    if var_type== 'nonim':
        varsfilename = '/media/raw_data/UKB_2/ANALYSIS/non_imaging_UKB2_Janus.csv'
    if var_type == 'Svars':
        varsfilename = '/media/raw_data/UKB_2/ANALYSIS/Svars_UKB2_Janus.csv'

    no_subj = len(IDs_list)
    IDs_list_mod = [IDs_list[i][1:] for i in range(no_subj)] #Removing session-number prefix to obtain raw ID
    variabs_temp = []

    with open(varsfilename, 'r', encoding='UTF8') as covarcsv:
        csvreader = csv.reader(covarcsv)
        ctrl = 0
        for row in csvreader:
            if ctrl == 0:
                covar_names = row
                covar_inds = [covar_names.index(varnames[i]) for i in range(len(varnames))]

                ctrl = 1
            else:
                if str(row[0]) in IDs_list_mod:
                    row_vals = [int(row[0])]
                    row_vals.extend([row[i] for i in covar_inds])
                    variabs_temp.append(row_vals)
    
    variabs = np.array(variabs_temp, dtype=object)
        
    var_vals = np.zeros([no_subj, len(varnames)])
    IDs_array_mod = np.array(IDs_list_mod, dtype = int)
    variabs_IDs_list = [variabs[i,0] for i in range(variabs.shape[0])]

    for i in range(no_subj):
        subj_ind = variabs_IDs_list.index(IDs_array_mod[i]) #np.where(variabs[:,0] == IDs_array[i])[0]
        # for k in range(len(varnames)):
            # covar_nam = varnames[k]
            # covar_ind = covar_names.index(covar_nam)+1 ###OBSOBSOBSOBSOBOBSOBS
            # print('CHECK THIS')
        var_vals[i,:] = variabs[subj_ind, 1:]
    
    return var_vals

def get_varnames_UKB(var_type='basic'):
    #Reads variable names from the chosen variable file
    #var_type: Choice of variable file. Can be 'basic', 'Svars', or 'nonim'
    if var_type == 'basic':
        filename = '/media/raw_data/UKB_2/ANALYSIS/basicbehaviourals_varnames_UKB2_Janus.csv'
    elif var_type == 'Svars':
        filename = '/media/raw_data/UKB_2/ANALYSIS/Svars_varnames_UKB2_Janus.csv'
    elif var_type == 'nonim':
        filename = '/media/raw_data/UKB_2/ANALYSIS/non_imaging_varnames_UKB2_Janus.csv'
    else:
        raise Exception('Wrong specification of filename in function get_varnames_UKB')

    with open(filename, 'r', encoding='UTF8') as f:
        csvreader = csv.reader(f)
        ctrl = 0
        for row in csvreader:
            var_names = row

    return var_names
    

def get_age_standard(IDs_list, age_trans_type, dataset):
    age_var = read_behavioural_csv(IDs_list, covars=['age'], dataset=dataset)
    age_stand = standardize_age(age_var, age_trans_type, dataset = dataset)
    return age_stand


def standardize_age(age_var, transf_type, dataset):
    #Transforms the age variable linearly to the interval [-1,1]
    #Paramters:
    #   age_var: Array of age values
    #   transf_type: 0, 1, or 2. 0: No transformation. 1: Linearly to [0,1]. 2: Linearly to [-1,1]
    #Returns:
    #   age_transf: Array of transformed age values. Same length as age_var.

    IDs_all = get_IDs(dataset=dataset)

    age_all = read_behavioural_csv(IDs_all, covars = ['age'], dataset=dataset)

    if np.ndim(age_var) == 2:
        age_var = np.squeeze(age_var)
    if transf_type != 0:
        age_range = np.array([np.min(age_all), np.max(age_all)])
        age_fact = (age_range[1]-age_range[0])/transf_type
        age_div = age_var/age_fact
        age_trans = age_div-np.min((age_all/age_fact))-(transf_type-1)
    else:
        age_trans = age_var
    return age_trans

def get_diagnosis_subjects(diag_codes = ['F', 'G'], dataset = 'UKB'):

    if not type(diag_codes) == list:
        raise Exception('Could not load diagnosis subjects. Diagnosis codes must be provided in a list.')

    basevarname = 'Diagnoses - ICD10 ('
    var_names = get_varnames_UKB('nonim')

    varnameinds_alldiag = []
    for diagn in diag_codes:
        varname_inds = []
        if type(diagn) is not list:
            varnam = basevarname + diagn
            for k in range(len(var_names)):
                test1 = var_names[k].find(varnam)
                if test1>=0:
                    varname_inds.append(k)
        elif type(diagn) == list:
            for k1 in range(len(diagn)):
                diagn_el = diagn[k1]
                varnam = basevarname + diagn_el
                for k in range(len(var_names)):
                    test1 = var_names[k].find(varnam)
                    if test1>=0:
                        varname_inds.append(k)

        varnameinds_alldiag.append(varname_inds)

    IDs_list = get_IDs(dataset=dataset)
    no_subj = len(IDs_list)
    IDs_new_mod = [IDs_list[i][1:] for i in range(no_subj)] #Removing session-number prefix to obtain raw ID

    varsfilename = '/media/raw_data/UKB_2/ANALYSIS/non_imaging_UKB2_Janus.csv'

    diaggroup_IDs = [[] for i in range(len(diag_codes))]

    with open(varsfilename, 'r', encoding='UTF8') as covarcsv:
        csvreader = csv.reader(covarcsv)
        ctrl = 0
        for row in csvreader:
            if ctrl == 0:
                covar_names = row
                covar_inds = [[covar_names.index(var_names[varnameinds_alldiag[k][i]]) for i in range(len(varnameinds_alldiag[k]))] for k in range(len(varnameinds_alldiag))]

                ctrl = 1
            else:
                if str(row[0]) in IDs_new_mod:
                    for k in range(len(varnameinds_alldiag)):
                        row_vals = np.array([float(row[i]) for i in covar_inds[k]])
                    
                        if np.any(row_vals):
                            ID_ind = IDs_new_mod.index(str(row[0]))
                            diaggroup_IDs[k].append(IDs_list[ID_ind])

    return diaggroup_IDs


def combine_diagnosis_subjects(IDs_diag):
    if len(IDs_diag)>0 and type(IDs_diag)==list:
        test_IDs_diag = np.all([type(IDs_diag[i])==list for i in range(len(IDs_diag))])
    
    
        if test_IDs_diag:
            IDs_combined = set(IDs_diag[0])
            for i in range(1,len(IDs_diag)):
                IDs_combined = set(IDs_combined).union(set(IDs_diag[i]))
        else:
            IDs_combined = IDs_diag
            warnings.warn('Could not combine diagnosis IDs. Please provide a list whose elements are lists.')
    else:
        raise Exception('Could not combine diagnosis IDs. Wrong variable type.')

    return IDs_combined

###Time series data

def load_dr_data(IDs, no_parc = 100, remove_badparc = 1, dataset = 'UKB', dataversion = 'new', subparc = []):
    """Load data for a list of IDs and store them in a dictionary.
    
    Parameters
    ----------
    IDs: IDs of fMRI subjects. For testing purposes loads 2 subjects per default.
    no_parc: Number of parcels. In UKB-data we have 25 or 100 parcels.
    
    Returns:
    --------
    data_coll: Dictionary with key:subject ID and value:array-like of shape (no_timepoint, no_parcels)
    """
    
    # no_IDs = len(IDs)
    data_coll = {}


    data_coll_list = load_dr_data_list(IDs, no_parc, remove_badparc, dataset, dataversion=dataversion, subparc = subparc)

    for i in range(len(IDs)):
        # foldernum = f'{i[0]}_{i[1]}/'
        # filename = userfolder + 'dr_files/IDs_' + foldernum + str(i) + '/fMRI/rfMRI_' + str(no_parc) + '.dr/dr_stage1.txt'
        # # data = pd.read_csv(filename, sep='\s+', header=None)
        ID = IDs[i]
        data = data_coll_list[i]
        data_coll.update({str(ID): data})

    return data_coll


def load_dr_data_standard(IDs, no_parc = 100, ts_stand_type=2, remove_startinds = 1, remove_badparc = 1, dataset = 'UKB', dataversion = 'new', subparc = []):
    
    if dataset == 'HCP':
        remove_startinds = 0
        remove_badparc = 0
    
    data_coll = load_dr_data(IDs, no_parc, remove_badparc=remove_badparc, dataset=dataset, dataversion=dataversion, subparc = subparc)
    data_coll_t = truncate_ts_all(data_coll, remove_firstinds=remove_startinds)
    data_coll_stand = standardize_ts(data_coll_t, ts_stand_type)
    return data_coll_stand


def load_dr_data_list(IDs, no_parc = 100, remove_badparc = 1, dataset = 'UKB', dataversion = 'new', subparc = []):
    """Load data for a list of IDs and store them in a dictionary.
    
    Parameters
    ----------
    IDs: IDs of fMRI subjects. For testing purposes loads 2 subjects per default.
    no_parc: Number of parcels. In UKB-data we have 25 or 100 parcels.
    
    Returns:
    --------
    data_coll_list: List of length n_subj with each element being array-like of shape (no_timepoint, no_parcels)
    """

    if not remove_badparc and len(subparc)>0:
        warnings.warn('WARNING: A subparcellation was chosen without removing artefact components. Make sure the numbering is correct.')

    
    no_IDs = len(IDs)
    data_coll_list = []
    if dataset == 'UKB': #Good ICA component indices can be found in UK Biobanks online ressources
        if no_parc == 25:
            good_parc_inds_init = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])-1
        elif no_parc == 100:
            good_parc_inds_init = np.array([2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 52, 53, 57, 58, 60, 63, 64, 93])-1
    else:
        good_parc_inds_init = np.arange(no_parc)

    if len(subparc)>0:
        good_parc_inds = good_parc_inds_init[subparc]
    else:
        good_parc_inds = good_parc_inds_init


    if dataset == 'UKB':
        if dataversion == 'old':
            userfolder = '/media/raw_data/UK_Biobank/'
        elif dataversion == 'new':
            userfolder = '/media/raw_data/UKB_2/IMAGING/fMRI/subjectsAll/'
    elif dataset == 'HCP':
        print('Needs to specify Linux path to HCP data')

    for i in IDs:
        if (dataset == 'UKB') or (dataset == 'UKB_surf'):
            if dataset == 'UKB':
                if dataversion == 'old':
                    foldernum = f'{i[0]}_{i[1]}/'
                    filename = userfolder + 'dr_files/IDs_' + foldernum + str(i) + '/fMRI/rfMRI_' + str(no_parc) + '.dr/dr_stage1.txt'
                    # data = pd.read_csv(filename, sep='\s+', header=None)
                elif dataversion == 'new':
                    filename = userfolder + str(i) + '/rfMRI_' + str(no_parc) + '.dr/dr_stage1.txt'
            elif dataset == 'UKB_surf':
                filename = '/media/raw_data/UKB_2/IMAGING/fMRI/subjectsAll/' + str(i) + f'/surf_fMRI/bb.rfMRI.DR_d{no_parc}.timecourse.txt'

            data_init = np.genfromtxt(filename, dtype = float)

            if remove_badparc:
                data = data_init[:,good_parc_inds]
            else:
                data = data_init
        elif dataset == 'HCP':
            filename = userfolder + i +f'/fMRI_all/fMRI_{no_parc}/fMRI_ts.txt'
            data = np.genfromtxt(filename, dtype = float)

        data_coll_list.append(data)

    return data_coll_list

def load_data_stand_list(IDs, no_parc = 100, ts_stand_type=2, remove_startinds = 1, remove_badparc = 1, dataset = 'UKB', subparc = []):
    data_coll_stand = load_dr_data_standard(IDs, no_parc, ts_stand_type, remove_startinds, remove_badparc, dataset=dataset, subparc = subparc)
    data_coll_stand_list = [data_coll_stand[i] for i in IDs]
    return data_coll_stand_list

def make_data_stand_list(data_coll_stand):
    """
    Creates list of time series matrices from already loaded data in dictoinary.
    """
    data_coll_stand_list = []
    for key in data_coll_stand.keys():
        data_coll_stand_list.append(data_coll_stand[key])
    return data_coll_stand_list

def load_data_stand_calc_FC(IDs, n_parc = 100, ts_stand_type=2, remove_startinds = 1, remove_badparc = 1, dataset = 'UKB', FC_type = 'Pearson', subparc = []):
    Y_dat = load_data_stand_list(IDs = IDs, no_parc = n_parc, ts_stand_type=ts_stand_type, remove_startinds = remove_startinds, remove_badparc = remove_badparc, dataset = dataset, subparc = subparc)
    FC_mat_all = calc_FCmatrix_all_fromlist(Y_dat, FC_type=FC_type) 

    return FC_mat_all

###Load and save

def save_data_pickle(var_names, variabs, filename):
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
        raise Exception('Loaded data varaible of incompatible format.')

    return var_names, var_dic

