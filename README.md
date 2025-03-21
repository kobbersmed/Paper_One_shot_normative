# Important info about the scripts

These scripts were used to fit the FUNCOIN model and do all formal analysis for the manuscript "One-shot normative modelling of whole-brain functional connectivity".

## Data availability
The data presented are from UK Biobank, which is only available upon application to the UK Biobank. Even though the reader may have access to the UK Biobank rsfMRI data, the functions we used for loading the data for the paper do not work, because 1) subject IDs are different for each data application, so anyone else with access to the data will need to use that project's specific subject IDs; 2) the functions refer to a specific local path, where our copy of the UK Biobank data was stored; and 3) the functions read the specific format the data was saved in at our end.




## Running the scripts
To facilitate reproducability, all functions loading data are modified so they can easily be adapted for the reader's local data copy. In each function, it is specified what the function should return and in which format. It is important to keep the `**kwargs` in the function definition, since this will make it ignore all variables originally given as parameters to these functions. All functions that need to be modified can be found in `io_funcs.py` and include:
* `prepare_run_funcoin(**kwargs)`: Called in `funcoin_runscripts.py -> funcoin_runnew_splittest_par()`
* `funcoin_prepare_from_IDslist(**kwargs)`: Called in `funcoin_runscripts.py -> run_funcoin_method` and `funcoin_runscripts.py -> funcoin_run_normative_UKB_diag`
* `read_behavioural_csv(data_type = '', **kwargs)`: Called in `funcoin_runscripts.py -> funcoin_run_normative_UKB_diag`. The parameter `data_type` specifies, if training, testing, or diagnosis data needs to be loaded, and is inputted appropriately whenever the function is calles in the script.
* `load_data_stand_calc_FC(data_type = '', **kwargs)`: Called in `LinReg_normative.py -> run_LinReg_normative`. The parameter `data_type` specifies, if training, testing, or diagnosis data needs to be loaded, and is inputted appropriately whenever the function is calles in the script.


After modifying the above functions to appropriately load the data, the script `paper_pipeline.py` runs the model fitting and analysis for the paper.

For transparency, all the original data loading functions (i.e. the functions actually used for generating the results for the paper) are included in `io_funcs_original.py`. They cannot be run without access to our data copy on our servers.

If the data variables are provided for each of the 4 functions, the scripts will run and return result files containing all result variables in a python dictionary saved with `pickle` and loadable with the function `io_funcs.py -> load_data_pickle(filename)`


## Important notes about data format
All necessary information about the return variables of the functions are provided in the function definition. Special care should be taken with the data returned by `prepare_run_funcoin(**kwargs)`, `funcoin_prepare_from_IDslist(**kwargs)`, and `read_behavioural_csv(data_type = '', **kwargs)`:
*  `prepare_run_funcoin(**kwargs)` and `funcoin_prepare_from_IDslist(**kwargs)`: In the X_dat variable (matrix with covariate information for all subjects), the age variable should be standardized to the interval [0,1] by linear transformation. This can be done with `io_funcs.py -> standardize_age(age_var, 1)`. The time series data in Y_dat should be normalized so each of the p brain regions have mean 0 and variance 1.
*  `read_behavioural_csv(data_type = '', **kwargs)`: Returns a numpy array with sex and age information about the subjects. In this case, the age variable is cronological age in years, i.e. not standardized in any way. 
