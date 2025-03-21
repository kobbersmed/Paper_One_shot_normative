from io_funcs import save_data_pickle, get_new_IDs_healthy, load_data_pickle
import numpy as np
import os


def gen_IDs_training(n_subj, datpath, set_seed, dataset):

    savefilename = datpath + f'IDs_list_training_random_{n_subj/1000}k'
    print(savefilename)
    checkfile = os.path.isfile(savefilename)

    if not checkfile:
        print('IDs_list file not found. Generating list and saving file with training IDs.')
        IDs_list = get_new_IDs_healthy(n_subj, random_io=1, set_seed=set_seed, dataset=dataset)

        var_names = ['IDs_list', 'set_seed']
        variabs = [IDs_list, set_seed]

        save_data_pickle(var_names, variabs, savefilename)
    else:
        print('Found file with training IDs and loaded the list.')
        var_names, var_dic = load_data_pickle(savefilename)
        IDs_list = var_dic['IDs_list']


    return IDs_list, savefilename