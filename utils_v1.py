import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
from sklearn import mixture
import math as m
from scipy import interpolate

# Function to compute numerical gradient using central finite difference
def gradientFiniteDifferent(func,theta,delta=1E-4):
    n = np.size(theta)
    grad = np.zeros((n))
    for i in range(n):
        theta_p=np.copy(theta)
        theta_m=np.copy(theta)
        theta_p[i]=theta_p[i]+delta
        theta_m[i]=theta_m[i]-delta
        f_plus = func(tf.constant(theta_p,dtype=tf.float32)).numpy()
        f_minus = func(tf.constant(theta_m,dtype=tf.float32)).numpy()
        grad[i] = (f_plus-f_minus)/(2*delta)
    return grad

# Numerically finding the icdf values for a distribution whos analytical CDF is specified
def icdf_numerical(u,cdf_funct,lb,ub):
    # setting up the numerical method (Chandrupatla root finding algorithm) to find icdf
    obj_func = lambda x: cdf_funct(x) - u
    # finding the roots
    x = tfp.math.find_root_chandrupatla(obj_func,low=lb,high=ub)[0]
    return x


def GMM_best_fit(samples,max_ncomp=10, print_info=False):
    lowest_bic = np.infty
    bic = []
    for n_components in range(max_ncomp):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components+1,covariance_type='full',max_iter=200,n_init=5)
        gmm.fit(samples)
        if print_info:
            print('Fittng a GMM on samples with %s components: BIC=%f'%(n_components,gmm.bic(samples)))
        bic.append(gmm.bic(samples))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm    
    return best_gmm

# Standardize GMM parameters
def standardize_gmm_params(alphas,mus,covs):
    weighted_mus = tf.linalg.matvec(tf.transpose(mus),alphas)
    new_mus = mus - weighted_mus
    variances = tf.linalg.diag_part(covs)
    scaling_vec = tf.linalg.matvec(tf.transpose(new_mus**2+variances),alphas)
    scaling_matrix = tf.linalg.diag(1/(scaling_vec**0.5))
    new_mus = tf.linalg.matmul(new_mus,scaling_matrix)
    new_covs = tf.linalg.matmul(covs,scaling_matrix**2)
    return alphas,new_mus,new_covs


def vec2gmm_params(n_dims,n_comps,param_vec):
    num_alpha_params = n_comps
    num_mu_params = n_comps*n_dims
    num_sig_params = int(n_comps*n_dims*(n_dims+1)*0.5)
    logit_param, mu_param, chol_param = tf.split(param_vec,[num_alpha_params,num_mu_params,num_sig_params])
    mu_vectors = tf.reshape(mu_param, shape=(n_comps,n_dims))
    chol_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    cov_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    for k in range(n_comps):
        start_idx = tf.cast(k*(num_sig_params/n_comps),tf.int32)
        end_idx = tf.cast((k+1)*(num_sig_params/n_comps),tf.int32)
        chol_mat = tfb.FillScaleTriL(diag_bijector=tfb.Exp()).forward(chol_param[start_idx:end_idx])
        cov_mat = tf.matmul(chol_mat,tf.transpose(chol_mat))
        chol_mat_array = chol_mat_array.write(k,chol_mat) 
        cov_mat_array =  cov_mat_array.write(k,cov_mat) 
        
    chol_matrices = chol_mat_array.stack()
    cov_matrices = cov_mat_array.stack()     
    return [logit_param,mu_vectors,cov_matrices,chol_matrices]

def gmm_params2vec(n_dims,n_comps,alphas,mu_vectors,cov_matrices):
    # now gathering all the parameters into a single vector
    param_list = []
    param_list.append(np.log(alphas))
    param_list.append(tf.reshape(mu_vectors,-1))
    for k in range(n_comps):
        chol_mat = tf.linalg.cholesky(cov_matrices[k])
        param_list.append(tfb.FillScaleTriL(diag_bijector=tfb.Exp()).inverse(chol_mat))
    param_vec = tf.concat(param_list,axis=0)
    return param_vec