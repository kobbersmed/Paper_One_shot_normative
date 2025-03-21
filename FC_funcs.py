import numpy as np
from scipy.linalg import logm, fractional_matrix_power
import matplotlib.pyplot as plt
import warnings

def test_matrixdef(mat):
    #Tests definitenes of a real (symmetric) matrix
    #----------
    #Parameters:
    #   mat : array-like of shape pxp
    #Outputs:
    #   def_stat: -1 if negative definite, 1 if positive definite, -0.1 if SND, 0.1 if SPD, 0 if indefinite
    #----------

    #Test if matrix is symmetric:
    sym_stat = test_matrixsym(mat)

    if not sym_stat:
        print("Warning: Matrix is not symmetric")
        return float('NaN')

    eigvals = np.linalg.eigvals(mat)
    n_eigs = len(eigvals)
    test_pos = sum(eigvals>0)

    if test_pos == len(eigvals):
        def_stat = 1
    elif sum(eigvals >= 0) == n_eigs:
        def_stat = 0.1
    elif sum(eigvals < 0) == n_eigs:
        def_stat = -1
    elif sum(eigvals <= 0) == n_eigs:
        def_stat = -0.1
    else:
        def_stat = 0

    return def_stat


def test_matrixsym(mat, rtol_inp=1e-5, atol_inp=1e-8):
    #Tests symmetry of a square matrix
    #Parameters:
    #   mat: array-like of shape pxp
    # 
    #Returns:
    #   sym_stat: True if symmetric, False if non-symmetric
    
    sym_stat = np.all(abs(mat-mat.T)<1e-6)
    # sym_stat = np.allclose(mat, mat.T, rtol = rtol_inp, atol = atol_inp)
    return sym_stat



def calc_corrmatrix(data, cor_partial = False):
    """Computes Pearson correlation matrix given timeseries from parcels.
    
    Parameters:
    -----------
    data: Array-like of shape (n_t, n_parcels).
    
    Returns:
    --------
    cormatrix: Array-like of shape (n_parcels, n_parcels)
    """
    if not cor_partial:
        cormatrix = np.corrcoef(data, rowvar = False)
        np.fill_diagonal(cormatrix, 1) #Set all diagonal values to 1. 
        
    return cormatrix
    #Implement partial correlation later

def calc_FCmatrix(data, type = 'Pearson'):
    """Computes FC matrix based on  either covariance, Pearson corr., Spearman corr, or partial correlation
    
    Parameters:
    -----------
    data: Array-like of shape (n_t, n_parcels).
    type: 'Pearson', 'Spearman', 'Partial', or 'Covariance'
    
    Returns:
    --------
    cormatrix: Array-like of shape (n_parcels, n_parcels)
    """

    if type == 'Pearson':
        FCmatrix = calc_corrmatrix(data)
    elif type == 'Covariance':
        FCmatrix = np.cov(data, rowvar = False)
    elif type == 'CovarianceBias':
        FCmatrix = np.cov(data, rowvar = False, bias=True)
    elif type == 'PartialCorr':
        corrmat = calc_corrmatrix(data)
        cor_inv = np.linalg.inv(corrmat)
        FCmatrix_init = np.zeros_like(corrmat)
        n_parc = corrmat.shape[0]
        for i in range(n_parc):
            FC_vec = [-cor_inv[i,j]/(np.sqrt(cor_inv[i,i]*cor_inv[j,j])) for j in range(i+1,corrmat.shape[0])]
            FCmatrix_init[i,i+1:] = FC_vec
        FCmatrix = FCmatrix_init + FCmatrix_init.T + np.identity(n_parc)
    elif type == 'invcorr':
        reg_const = 0.01
        FCmatrix_init = calc_corrmatrix(data)
        FCmatrix = np.linalg.inv(FCmatrix_init)
    elif type == 'Spearman':
        warnings.warn('WARNING: Spearman for connectivity analysis is not implemented.')
        FCmatrix = float('NaN')

    return FCmatrix
    #Implement partial correlation later

def make_FCvec(FCmatrix, incl_diag = False):
    #Creates a vector of FC edges (excluding diagonal)
    #Parameters:
    #   FCmatrix: Array-like of shape pxp
    #Returns:
    #   FCvec: Array of length p*(p-1)/2
    FCmatrix_triu = np.triu(FCmatrix,1-incl_diag)
    FCvec_full = FCmatrix_triu.flatten()
    FCvec = np.delete(FCvec_full, FCvec_full==0)

    return FCvec

def vectorise_triu_matrix(FCmat, incl_diag = False):
    n_parc = FCmat.shape[0]
    triu_inds = np.triu_indices(n_parc, 1 - incl_diag)
    FC_vec = FCmat[triu_inds]

    return FC_vec

def trans_FCvec_to_mat(FCvec, incl_diag = False):
    
    n_edge = len(FCvec)

    if not incl_diag:
        n_parc = round((1+np.sqrt(1+4*2*n_edge))/2) #Solving quadratic equation: p^2-p-2n = 0
    if incl_diag:
        n_parc = round((-1+np.sqrt(1+4*2*n_edge))/2) #Solving quadratic equation: p^2+p-2n = 0
    triu_inds = np.triu_indices(n_parc, 1 - incl_diag)

    FCmat = np.zeros([n_parc,n_parc])

    FCmat[triu_inds] = FCvec
    FCmat = FCmat + FCmat.T + np.identity(n_parc)

    return FCmat


def calc_FCmatrix_all(IDs_list, data_coll, FC_type='Pearson'):
    """Computes FC matrix based on either covariance, Pearson corr., Spearman corr, or partial correlation
    
    Parameters:
    -----------
    data_coll: Dictionary with key:subject ID and value:array-like of shape (no_timepoint, no_parcels)
    type: 'Pearson', 'Spearman', 'Partial', or 'Covariance'
    
    Returns:
    --------
    FCmatrix_all: Array-like of shape (n_subjects, n_edges)
    """
    # IDs_list = list(data_coll.keys()) #IDs_list is now an input variable to make sure that ordering is preserved
    n_subj = len(IDs_list)
    data_first = data_coll[IDs_list[0]]
    n_parcels = data_first.shape[1]

    FCmatrix_all = np.zeros([n_subj, round(n_parcels*(n_parcels-1)/2)]) #size n_subj X [no og edges]

    for i in range(len(IDs_list)):
        ID = IDs_list[i]
        data = data_coll[ID]
        FCmatrix = calc_FCmatrix(data, FC_type)
        FCvec = vectorise_triu_matrix(FCmatrix)
        FCmatrix_all[i,:] = FCvec

    return FCmatrix_all

def calc_FCmatrix_all_fromlist(data_list, FC_type='Pearson'):
    """Computes FC matrix based on either covariance, Pearson corr., Spearman corr, or partial correlation
    
    Parameters:
    -----------
    data_list: List of array-like time series data of shape (no_timepoint, no_parcels)
    type: 'Pearson', 'Spearman', 'Partial', or 'Covariance'
    
    Returns:
    --------
    FCmatrix_all: Array-like of shape (n_subjects, n_edges)
    """
    # IDs_list = list(data_coll.keys()) #IDs_list is now an input variable to make sure that ordering is preserved
    n_subj = len(data_list)
    data_first = data_list[0]
    n_parcels = data_first.shape[1]

    FCmatrix_all = np.zeros([n_subj, round(n_parcels*(n_parcels-1)/2)]) #size n_subj X [no og edges]

    for i in range(len(data_list)):
        data = data_list[i]
        FCmatrix = calc_FCmatrix(data, FC_type)
        FCvec = vectorise_triu_matrix(FCmatrix)
        FCmatrix_all[i,:] = FCvec

    return FCmatrix_all

def calc_FCmatrix_all_list(IDs_list, data_coll, FC_type='Pearson'):
    FCmatrices_list = []
    for ID in IDs_list:
        data = data_coll[ID]
        FCmat = calc_FCmatrix(data, type = FC_type)
        FCmatrices_list.append(FCmat)
    
    return FCmatrices_list
    
def calc_FCmatrix_all_listtolist(Y_dat_list, FC_type='Pearson'):
    FCmatrices_list = [calc_FCmatrix(Y_dat_list[i], type = FC_type) for i in range(len(Y_dat_list))]

    return FCmatrices_list

def corr_to_z(cormatrix):
    """Computes the Fisher r-z transformation. Transforms a matrix of correllations to approx. normally distributed values for statistical comparison and normative modelling.
    
    Parameters:
    -----------
    cormatrix: Array-like. Usually of shape (no_parcels, no_parcels) but works with any shape (e.g. matrix of shape (n_subjects, n_edges))
    
    Returns:
    --------
    zMatrix: Array-like of same shape as cormatrix. Matrix of z values from FC matrix. Entries with correlation 1 is set to NaN.
    """
    
    cormatrix_NaNs = np.where(cormatrix!=1, cormatrix, float('NaN'))
    zMatrix = 0.5*np.log((1+cormatrix_NaNs)/(1-cormatrix_NaNs))
    
    return zMatrix

def calc_matrix_dotprod(MatrixA, MatrixB):
    if type(MatrixA) == np.ndarray and type(MatrixB) == np.ndarray:
        vec_A = np.matrix.flatten(MatrixA)
        vec_B = np.matrix.flatten(MatrixB)
    else:
        vec_A = MatrixA
        vec_B = MatrixB
    
    mat_dot = np.dot(vec_A,vec_B)
    return mat_dot

def calc_matrix_dist(MatrixA, MatrixB, dist_type = 'euclidian'):
    vec_A = np.matrix.flatten(MatrixA)
    vec_B = np.matrix.flatten(MatrixB)
    if dist_type == 'euclidian':
        vec_dif = vec_A-vec_B
        mat_dist = np.sqrt(np.dot(vec_dif, vec_dif))
    elif dist_type == 'correlation':
        vecA_mean = np.mean(vec_A) 
        vecB_mean = np.mean(vec_B)
        vecA_sd = np.std(vec_A)
        vecB_sd = np.std(vec_B)
        r_vecs = np.sum([(vec_A[i]-vecA_mean)*(vec_B[i]-vecB_mean) for i in range(len(vec_A))])/(vecA_sd*vecB_sd)
        mat_dist = 1-r_vecs
    elif dist_type == 'riemannian':
        matA_power = fractional_matrix_power(MatrixA, -0.5)
        mat_prod = matA_power @ MatrixB @ matA_power
        mat_dist = np.sqrt(np.sum(np.log(np.linalg.eigvals(mat_prod))**2))

    return mat_dist


def calc_matrix_dist_list(mat_list, mat_ref, dist_type = 'euclidian'):
    mat_dists = np.zeros(len(mat_list))
    for i in range(len(mat_list)):
        mat_dist = calc_matrix_dist(mat_list[i], mat_ref, dist_type = 'euclidian')
        mat_dists[i] = mat_dist

    return mat_dists


def calc_ref_matrix(Cov_matrix_list, ref_type= 'euclidian'):

    if ref_type == 'euclidian':
        matrix_sum = Cov_matrix_list[0]
        for i in range(1,len(Cov_matrix_list)):
            matrix_sum = matrix_sum + Cov_matrix_list[i] 
        C_ref = (1/len(Cov_matrix_list))*matrix_sum
    elif ref_type == 'harmonic':
        matrix_inv_sum = np.linalg.inv(Cov_matrix_list[0])
        for i in range(1,len(Cov_matrix_list)):
            matrix_inv_sum = matrix_inv_sum + np.linalg.inv(Cov_matrix_list[i]) 
        C_ref = np.linalg.inv((1/len(Cov_matrix_list))*matrix_inv_sum)
    elif ref_type == 'log-euclidian':
        pass
    elif ref_type == 'riemann':
        pass
    elif ref_type == 'kullback':
        pass

    return C_ref



def project_SPD_tangentspace(MatrixA, C_ref_power):
    SPD_stat = test_matrixdef(MatrixA)

    if SPD_stat == 1:
        matrix_prod = C_ref_power @ MatrixA @ C_ref_power 
        tangent_mat = logm(matrix_prod)
    elif SPD_stat != 1:
        raise Exception('Matrix of a non-SPD matrix is now defined')
    return tangent_mat

def matrix_power_minhalf(C_ref):
    C_ref_power = fractional_matrix_power(C_ref, -0.5)
    return C_ref_power


def project_SPD_tangent_all(cov_matrix_list, C_ref):
    
    # C_ref = calc_ref_matrix(cov_matrix_list, ref_type = 'euclidian')
    C_ref_power = matrix_power_minhalf(C_ref)
    tan_mat_list = []
    for i in range(len(cov_matrix_list)):
        covmat = cov_matrix_list[i]
        tan_mat = project_SPD_tangentspace(covmat, C_ref_power)
        tan_mat_list.append(tan_mat)
    # C_ref_tan = project_SPD_tangentspace(C_ref, C_ref_power)
    
    return tan_mat_list

def plot_FC_matrix(cormatrix, title_str="FC matrix"):
    # np.fill_diagonal(cormatrix, 1)
    plt.imshow(cormatrix, vmin=-1, vmax=1)
    plt.title(title_str)
    plt.colorbar()




def vectorise_triu_matrix_list(FCmat_list, incl_diag = False):
    """Takes a list of matrices and returns the vectorized, upper triangular part (excluding diagonal) of each matrix in a matrix
    
    Parameters:
    -----------
    FCmat_list: List of symmetric, square matrices.
    
    Returns:
    --------
    FCmatrix_all: Array-like of shape (n_subjects, n_edges)
    """
    FCmat_all_init = [vectorise_triu_matrix(FCmat_list[i], incl_diag) for i in range(len(FCmat_list))]
    FCmat_all = np.array(FCmat_all_init)
    return FCmat_all
