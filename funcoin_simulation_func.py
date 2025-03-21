from funcoin import Funcoin as fcn
from io_funcs import save_data_pickle 
import numpy as np
import os
from scipy.stats import skewnorm


def funcoin_simulation_singlerun(filename_temp, nullcase, n_time, n_subj, n_bootstrap=500, CI_lvl = 0.05, max_comps = 2, betaLinReg = False, seed_sim= None, seed_bootstrap=None, gamma_noise = 0, skew_val=0):


    checkfile = os.path.isfile(filename_temp)
    if not checkfile:

        try:
            ind = int(filename_temp[-3:])
        except:
            try:
                ind = int(filename_temp[-2:])
            except:
                    ind = int(filename_temp[-1])
        print(f'Simulation {ind}')

        Y_sim, X_sim, _, gamma_mat_true, beta_mat_true, lambdas_subj = funcoin_simulate_data_paper(n_subj, n_time, nullcase, seed=seed_sim, gamma_noise=gamma_noise, seed_noise=seed_sim+10000, skew_val=skew_val)

        n_dir_used_bootstrap = [i for i in range(beta_mat_true.shape[1]) if np.sum(beta_mat_true[1:,i]!=0)]


        fcn_inst = fcn()

        fcn_inst.decompose_bootstrap(Y_sim, X_sim, n_bootstrap, max_comps, CI_lvl = CI_lvl, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, trace_sol = 0, seed_initial = None, betaLinReg = betaLinReg, seed_bootstrap=seed_bootstrap)
        
        gamma_mat = fcn_inst.gamma
        beta_mat = fcn_inst.beta
        beta_mats_bootstrap = fcn_inst.betas_bootstrap
        beta_mat_CI = fcn_inst.beta_CI
        dfd_values = fcn_inst.dfd_values_training

        var_names = ['gamma_mat', 'beta_mat', 'beta_mats_bootstrap', 'beta_mat_CI', 'X_sim', 'dfd_values']
        variabs = [gamma_mat, beta_mat, beta_mats_bootstrap, beta_mat_CI, X_sim, dfd_values]
        save_data_pickle(var_names, variabs, filename_temp)



def funcoin_simulate_data_paper(n_subj, n_time = 100, nullcase = False, seed=None, gamma_noise=0, seed_noise=None, skew_val = 0):

    rng = np.random.default_rng(seed=seed)
    X_cov = rng.binomial(1,0.5,n_subj)
    X_sim = np.array([np.ones(n_subj), X_cov]).T

    if nullcase:
        beta_mat = np.array([[5,4,1,-1,-2], [0,0,0,0,0]])
        lambdas_subj = np.array([rng.lognormal(beta_mat[0,:], 0.5) for i in range(n_subj)])
    else:
        beta_mat = np.array([[5,4,1,-1,-2], [0,-1,1,0,0]])
        lambdas_subj = np.array([np.exp(X_sim[i,:]@beta_mat) for i in range(n_subj)])

    p_model = beta_mat.shape[1]

    Gamma_mat = np.ones([p_model,p_model])*np.sqrt(0.019) + np.identity(p_model)*(-np.sqrt(0.743) - np.sqrt(0.019))
    Gamma_mat[0,:] = np.sqrt(0.2)
    Gamma_mat[:,0] = np.sqrt(0.2)

    

    if gamma_noise != 0:
        if skew_val == 0:
            rng_noise = np.random.default_rng(seed=seed_noise)
            Sigma_list = []
            for k in range(n_subj):
                noise_term = rng_noise.normal(0, gamma_noise, size=Gamma_mat.shape)
                new_gamma = (Gamma_mat+noise_term)
                for k3 in range(Gamma_mat.shape[1]):
                    new_gamma[:,k3] = new_gamma[:,k3]/np.linalg.norm(new_gamma[:,k3])
                Sigma_list.append(new_gamma@np.diag(lambdas_subj[k])@new_gamma.T)
        else:
            mean_theo = 0 + gamma_noise*(skew_val/(np.sqrt(1+skew_val**2)))*np.sqrt(2/np.pi)
            Sigma_list = []
            for k in range(n_subj):
                noise_term = skewnorm.rvw(a=skew_val, loc=0, scale=gamma_noise, size=Gamma_mat.shape)-mean_theo
                new_gamma = (Gamma_mat+noise_term)
                for k3 in range(Gamma_mat.shape[1]):
                    new_gamma[:,k3] = new_gamma[:,k3]/np.linalg.norm(new_gamma[:,k3])
                Sigma_list.append(new_gamma@np.diag(lambdas_subj[k])@new_gamma.T)
    else:
        Sigma_list = [Gamma_mat@np.diag(lambdas_subj[i])@Gamma_mat.T for i in range(n_subj)]   

    mean_sim = np.zeros(p_model)

    Y_sim = [rng.multivariate_normal(mean_sim, Sigma_list[i], n_time) for i in range(n_subj)]

    return Y_sim, X_sim, Sigma_list, Gamma_mat, beta_mat, lambdas_subj
