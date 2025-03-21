import time
from multiprocessing import Pool
import os
from funcoin_simulation_func import funcoin_simulation_singlerun, funcoin_simulate_data_paper
from io_funcs import *

n_sim=200
n_bootstrap=2
parallel_io = 1
max_comps = 5
CI_lvl = 0.05
n_pool = 8
nullcase = False
n_time = 100
n_subj = 100
seed_init = 3214

for skew_val in [0, 0.5, 1, 2]:
    for gamma_noise in [0, 0.01, 0.1, 0.5, 1]:
        for betaLinReg in [1,0]:#,True]:

            _,_, _, gamma_mat_true, beta_mat_true, _ = funcoin_simulate_data_paper(n_subj, n_time = n_time, nullcase = nullcase, gamma_noise = 0, skew_val=0)

            gamma_mats_all = []
            beta_mats_all = []
            beta_mats_bootstrap_all = []
            beta_mat_CI_all = []
            X_sim_all = []
            dfd_vals_all = []
            # for other_beta in [0,1]:
            #     for n_time in [50, 100, 500, 1000]:
            #         for n_subj in [50, 100, 500, 1000]:
            print(f'START. nullcase: {nullcase}, n_time: {n_time}, n_subj: {n_subj}, n_sim: {n_sim}, n_bootstrap: {n_bootstrap}, betaLinReg: {betaLinReg}, newSI!')

            timeA = time.time()

            datpath = './paper_data_mod/simstudy/'
            datpath_temp = './paper_data_mod/simstudy/temp/'
            filename = f'FUNCOIN_simulationstudy_nullcase{nullcase}_ntime{n_time}_nsubj{n_subj}_nsim{n_sim}_nboot{n_bootstrap}_maxcomps{max_comps}_betaLinReg{betaLinReg}_SEEDEDsimbs_gammanoise{gamma_noise}_skewval{skew_val}'

            savefilename = datpath + filename

            check_savefile = os.path.isfile(savefilename)
            if check_savefile:
                print(savefilename + ' found')

            if not check_savefile:
                if not parallel_io:
                    for i6 in range(n_sim):
                        seed_sim = seed_init+i6
                        seed_bootstrap = seed_init-i6
                        filename_temp = datpath_temp + filename + f'_temp{i6}'
                        funcoin_simulation_singlerun(filename_temp, nullcase, n_time, n_subj, n_bootstrap=n_bootstrap, CI_lvl = CI_lvl, max_comps = max_comps, betaLinReg = betaLinReg, seed_sim=seed_sim, seed_bootstrap=seed_bootstrap, gamma_noise=gamma_noise, skew_val=skew_val)

                elif parallel_io:
                    args_par = []
                    for i6 in range(n_sim):
                        seed_sim = seed_init+i6
                        seed_bootstrap = seed_init-i6
                        filename_temp = datpath_temp + filename + f'_temp{i6}'
                        args_par.append((filename_temp, nullcase, n_time, n_subj, n_bootstrap, CI_lvl, max_comps, betaLinReg, seed_sim, seed_bootstrap, gamma_noise, skew_val))

                    if __name__ == '__main__':
                        with Pool(processes=n_pool) as pool:
                            pool.starmap(funcoin_simulation_singlerun, args_par) 
                    
                for i5 in range(n_sim):
                    filename_temp = datpath_temp + filename + f'_temp{i5}'

                    var_names, var_dic = load_data_pickle(filename_temp)

                    gamma_mat = var_dic['gamma_mat']
                    beta_mat = var_dic['beta_mat']
                    beta_mats_bootstrap = var_dic['beta_mats_bootstrap']
                    beta_mat_CI = var_dic['beta_mat_CI']
                    X_sim = var_dic['X_sim']
                    dfd_values = var_dic['dfd_values']

                    gamma_mats_all.append(gamma_mat)
                    beta_mats_all.append(beta_mat)
                    beta_mats_bootstrap_all.append(beta_mats_bootstrap)
                    beta_mat_CI_all.append(beta_mat_CI)
                    X_sim_all.append(X_sim)
                    dfd_vals_all.append(dfd_values)

                variabs = [beta_mats_all, gamma_mats_all, beta_mats_bootstrap_all, beta_mat_CI_all, gamma_mat_true, beta_mat_true, X_sim_all, dfd_vals_all]
                var_names = ['beta_mats_all', 'gamma_mats_all', 'beta_mats_bootstrap_all', 'beta_mat_CI_all', 'gamma_mat_true', 'beta_mat_true', 'X_sim_all', 'dfd_vals_all']

                save_data_pickle(var_names, variabs, savefilename)


            check_savefile2 = os.path.isfile(savefilename)
            if check_savefile2:
                for i4 in range(n_sim):
                    filename_temp = datpath_temp + filename + f'_temp{i4}'
                    check_temp = os.path.isfile(filename_temp)
                    if check_temp:
                        os.remove(filename_temp)

            timeelaps = (time.time() - timeA)/60

            print(f'END. nullcase: {nullcase}, n_time: {n_time}, n_subj: {n_subj}, n_sim: {n_sim}, n_bootstrap: {n_bootstrap}, betaLinReg: {betaLinReg}')
            print(f'Time elapsed: {timeelaps} minutes')
