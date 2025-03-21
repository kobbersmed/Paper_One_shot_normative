import numpy as np
import os
from funcoin import Funcoin as fcn
from io_funcs import save_data_pickle


def funcoin_splittest_singlerun_func(X_dat, Y_dat, n_dir, rand_init = 1, n_init = 20 , seed_initial = None, set_seed_split_init=None, betaLinReg = 1, dataset='UKB', filename='test'):
    
    try:
        ind = int(filename[-3:])
    except:
        try:
            ind = int(filename[-2:])
        except:
                ind = int(filename[-1])

    ctrl = os.path.isfile(filename)

    if not ctrl:
        n_subj = len(Y_dat)
        n_per_split = round(n_subj/2)
        print(f'Test number {ind}')
        if set_seed_split_init:
            set_seed_split = set_seed_split_init + ind +1
            rng = np.random.default_rng(seed=set_seed_split)
        else:
            rng = np.random.default_rng()

        train_perm = rng.permutation(n_subj)
        split1_inds = train_perm[:n_per_split]
        split2_inds = train_perm[n_per_split:]

        X_dat1 = X_dat[split1_inds,:]
        X_dat2 = X_dat[split2_inds,:]
        Y_dat1 = [Y_dat[i] for i in split1_inds]
        Y_dat2 = [Y_dat[i] for i in split2_inds]

        fcn_inst1 = fcn()
        fcn_inst2 = fcn()

        fcn_inst1.decompose(Y_dat1, X_dat1, n_dir, rand_init = rand_init, n_init = n_init, seed_initial=seed_initial, betaLinReg=betaLinReg)
        fcn_inst2.decompose(Y_dat2, X_dat2, n_dir, rand_init = rand_init, n_init = n_init, seed_initial=seed_initial, betaLinReg=betaLinReg)
    
        beta_mat1 = fcn_inst1.beta
        gamma_mat1 = fcn_inst1.gamma
        beta_mat2 = fcn_inst2.beta
        gamma_mat2 = fcn_inst2.gamma

        dfd_vals_geom1 = fcn_inst1.dfd_values_training
        dfd_vals_geom2 = fcn_inst2.dfd_values_training


        variabs = [beta_mat1, beta_mat2, gamma_mat1, gamma_mat2, dfd_vals_geom1, dfd_vals_geom2, X_dat1, X_dat2]
        var_names = ['beta_mat1', 'beta_mat2', 'gamma_mat1', 'gamma_mat2', 'dfd_vals_geom1', 'dfd_vals_geom2', 'X_dat1', 'X_dat2']

        
        save_data_pickle(var_names, variabs, filename)



