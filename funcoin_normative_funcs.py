import numpy as np

def sort_subjectinds_oneyearagegroups(age_orig, min_age, max_age):
    #Returns indices og age_orig sorted in one-year agegroups.

    int_min = np.floor(min_age)
    int_max = np.ceil(max_age)
    agegroups_lowerbounds = np.arange(int(int_min), int(int_max))
    n_agegroups = len(agegroups_lowerbounds)

    agegroups_inds = [[i for i in range(len(age_orig)) if (age_orig[i]>= agegroups_lowerbounds[k] and age_orig[i]<agegroups_lowerbounds[k]+1)] for k in range(n_agegroups)]
    
    return agegroups_inds