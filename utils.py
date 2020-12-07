# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script with several utility functions.
"""

import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from sklearn import mixture
import math as m
from scipy import interpolate
from custom_bijectors import GMC_bijector, mv_GMM_icdf_bijector



# Defining a 2-d bijective transformation of the form y_1 = c0*x_1+c1, y_2 = c2*x_1^2 + c3*x_2 + c4
# This will be used to define our reference distribution, the samples from which will be used for density estimation
class Banana(tfb.bijector.Bijector):

   def __init__(self, coeff=[1.,1.,1.,1.,1.], validate_args=False, name='banana'):
        super(Banana, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          is_constant_jacobian=True,          
          name=name)
        self.coeff = coeff

   def _forward(self, x):        
        y = np.zeros(np.shape(x),dtype=np.float32)
        y[:,0] = self.coeff[0]*x[:,0]+self.coeff[1]
        y[:,1] = self.coeff[2]*x[:,0]**2 + self.coeff[3]*x[:,1] + self.coeff[4]
        return y

   def _inverse(self, y):
        n_rows = np.size(y,0)
        x0 = tf.TensorArray(dtype=tf.float32, size=n_rows, dynamic_size=False)
        x1 = tf.TensorArray(dtype=tf.float32, size=n_rows, dynamic_size=False)
        for i in np.arange(n_rows):
            x0 = x0.write(i , (y[i,0]-self.coeff[1])/self.coeff[0])
            x1 = x1.write(i , (y[i,1] - self.coeff[2]*((y[i,0]-self.coeff[1])/self.coeff[0])**2 - self.coeff[4])/self.coeff[3])

        x0 = x0.stack()
        x1 = x1.stack()

        x0 = tf.reshape(x0,shape=(n_rows,1))
        x1 = tf.reshape(x1,shape=(n_rows,1))
        x = tf.concat([x0,x1],axis=1)
        return x

   def _inverse_log_det_jacobian(self, y):
        return  -np.ones((np.size(y,0),1))*(np.log(np.abs(self.coeff[0])) + np.log(np.abs(self.coeff[3])))

   def _forward_log_det_jacobian(self, x):
        return np.ones((np.size(x,0),1))*(np.log(np.abs(self.coeff[0])) + np.log(np.abs(self.coeff[3])))

# Sample-based ELBO (Computing the ELBO when we have samples from the posterior)
def ELBO_from_samples(target_log_prob_fn,samples):
    # first fit a parametric distribution on the posterior samples
    gmm_dist = GMM_best_fit(samples, max_ncomp=6)    
    ELBO = tf.reduce_mean(target_log_prob_fn(samples.astype('float32'))) - np.mean(gmm_dist.score_samples(samples))
    return ELBO

# Sample-based Entropy
def Entropy_from_samples(target_log_prob_fn,samples):
    # first fit a parametric distribution on the posterior samples
    gmm_dist = GMM_best_fit(samples, max_ncomp=6)    
    entropy =  -np.mean(gmm_dist.score_samples(samples))
    return entropy


# Fitting a Gaussian Mixture Model on the samples. Number of components is identified automatically
def GMM_best_fit(samples,max_ncomp=10):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, max_ncomp)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,covariance_type='full',max_iter=200,n_init=5)
        gmm.fit(samples)
        print('Fittng a GMM on samples with %s components: BIC=%f'%(n_components,gmm.bic(samples)))
        bic.append(gmm.bic(samples))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm    
    return best_gmm

# univariate gmm cdf
def gmm_cdf(x,alpha,mu,var):
    sig = tf.math.sqrt(var)
    u=tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                          components_distribution=tfd.Normal(loc=mu,scale=sig)).cdf(x)
    return u     
# univariate gmm logpdf
def gmm_lpdf(x,alpha,mu,var):
    sig = tf.math.sqrt(var)
    lpdf = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                             components_distribution=tfd.Normal(loc=mu,scale=sig)).log_prob(x)
    return lpdf
# univariate gmm pdf
def gmm_pdf(x,alpha,mu,var):
    lpdf = gmm_lpdf(x,alpha,mu,var)
    return tf.math.exp(lpdf)

# multivariate gmm logpdf
def mv_gmm_lpdf(x_in,alpha,mu,chol_mat):
    ncomps = alpha.get_shape().as_list()[0]
    input_shape = x_in.get_shape().as_list()
    if len(input_shape) == 1:
        ndims = input_shape[0]
        x_in  = tf.reshape(x_in,shape=(1,ndims))
    elif len(input_shape) == 2:
        nsamps,ndims = input_shape  
    pi = tf.cast(m.pi,tf.float64)
    temp_array = tf.TensorArray(tf.float64,size=ncomps)
    for k in range(ncomps):
        temp_mat = tf.linalg.matmul(tf.linalg.inv(chol_mat[k,:,:]),tf.transpose(x_in - mu[k,:]))
        temp_mat = tf.transpose(temp_mat)
        log_det_chol = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_mat[k,:,:])))
        val = tf.math.log(alpha[k])-0.5*ndims*tf.math.log(2*pi) - log_det_chol - 0.5*tf.reduce_sum(temp_mat*temp_mat,axis=1)
        temp_array = temp_array.write(k,val)
    weighted_pdfs = tf.transpose(temp_array.stack())
    lpdf = tf.math.reduce_logsumexp(weighted_pdfs,axis=1)
    return lpdf, weighted_pdfs

# univariate normal pdf
def normal_pdf(x,mu,var):
    sig = tf.math.sqrt(var)
    lpdf = tfd.Normal(loc=mu,scale=sig).log_prob(x)
    return tf.math.exp(lpdf)

# multivariate normal pdf
def mvn_pdf(x,mu,covar):
    lpdf = mvn_lpdf(x,mu,covar)
    return tf.math.exp(lpdf)

# multivariate normal logpdf
def mvn_lpdf(x,mu,covar):
    lpdf = tfd.MultivariateNormalFullCovariance(loc=mu,covariance_matrix=covar).log_prob(x)
    return lpdf

# approximate inverse CDF of a univariate gmm
def gmm_icdf_approx(u,alpha,mu,var):
    # Clipping the values of u to be between 1E-16 and (1-1E-16) for erfinv to give bounded answer
    u = tf.where(u<1E-16, 1E-16*tf.ones_like(u,dtype=tf.float64), u)
    u = tf.where(u>(1-1E-16), (1-1E-16)*tf.ones_like(u,dtype=tf.float64), u)
    mu_agg = tf.reduce_sum(alpha*mu)
    var_agg = tf.reduce_sum(alpha*(var+tf.math.pow(mu,2)))-tf.math.pow(mu_agg,2)
    x = mu_agg + tf.math.sqrt(2*var_agg)*tf.math.erfinv(2*u-1)
    return x

# numerically defining the univariate gmm quantile function 
@tf.custom_gradient
def gmm_icdf(u,alpha,mu,var):
    n_points = tf.size(u)
    f = lambda xx: u - gmm_cdf(xx,alpha,mu,var)
    x_approx = gmm_icdf_approx(u,alpha,mu,var)
    x_approx = tf.reshape(x_approx,shape=(n_points,))
    sol = tfp.math. secant_root(objective_fn=f, initial_position=x_approx)
    x = sol[0]
    converged = tf.cast(tf.abs(sol[1])<1E-6,dtype=tf.float64) # identifying a binary vector denoting convergence of the secant method
    x = x*converged + x_approx*(1-converged) # replacing the inverse values with the analytical approximation when not converged
#     return x
    def grad(dy):
        # Calling  another python function to get the partial derivatives
        grad_alpha, grad_mu, grad_var = partial_deriv_gmm_icdf(x,alpha,mu,var)
        return tf.zeros(n_points,dtype=tf.float64), tf.linalg.matvec(grad_alpha,dy), tf.linalg.matvec(grad_mu,dy), tf.linalg.matvec(grad_var,dy)
    return x, grad

@tf.custom_gradient
def gmm_icdf_alt(u,alpha,mu,var):
    n_points = tf.size(u)
    n_bins = 10000
    x_min = tf.reduce_min(mu) - 5*tf.reduce_max(tf.sqrt(var))
    x_max = tf.reduce_max(mu) + 5*tf.reduce_max(tf.sqrt(var))
    x_ref = tf.linspace(x_min,x_max,n_bins)
    u_ref = gmm_cdf(x_ref, alpha, mu, var)  
    # using SciPy interpolation function (TF does not interpolation on unregular grids)
    x = tf.py_function(func=scipy_interp1d, inp=[u, u_ref, x_ref], Tout=tf.float64)
    x = tf.constant(x,dtype=tf.float64)
    # return x
    def grad(dy):
        # Calling  another python function to get the partial derivatives
        grad_alpha, grad_mu, grad_var = partial_deriv_gmm_icdf(x,alpha,mu,var)
        return tf.zeros(n_points,dtype=tf.float64), tf.linalg.matvec(grad_alpha,dy), tf.linalg.matvec(grad_mu,dy), tf.linalg.matvec(grad_var,dy)
    return x, grad

def scipy_interp1d(x,xref,yref,method='linear'):
    f = interpolate.interp1d(xref,yref,kind=method)
    return f(x)


def partial_deriv_gmm_icdf(z,alpha,mu,var):
    # getting the number of components
    ncomps = alpha.get_shape().as_list()[0]
    marg_gmm_pdf_val = gmm_pdf(z,alpha,mu,var)
    # gradent w.r.t gmc params
    grad_alpha_array = tf.TensorArray(tf.float64, size=ncomps)
    grad_mu_array = tf.TensorArray(tf.float64, size=ncomps)
    grad_var_array = tf.TensorArray(tf.float64, size=ncomps)
    for k in range(ncomps):
        v1 = -(0.5/marg_gmm_pdf_val)*(1+tf.math.erf((z-mu[k])/tf.math.sqrt(2*var[k])))
        v2 = alpha[k]*normal_pdf(z,mu[k],var[k])/marg_gmm_pdf_val
        v3 = v2 * ((z-mu[k])/(2*var[k])) 
        grad_alpha_array = grad_alpha_array.write(k, v1)
        grad_mu_array = grad_mu_array.write(k, v2 )
        grad_var_array = grad_var_array.write(k, v3 )
    return grad_alpha_array.stack(), grad_mu_array.stack(), grad_var_array.stack()

# Function to transform a vector to triangular matix (assumes a one dimensional vector)
def vector2Tril(x,diag_exp=True):
    n = x.get_shape().as_list()[0]
    # converting to a row vector
    x = tf.reshape(x,shape=(1,n))
    a,b,c =1,1,-2*n
    dim = int((-b+(b**2-4*a*c)**0.5)/(2*a))
    tril_mat = tf.TensorArray(tf.float64,size=dim)
    end_idx = 0
    for j in range(dim):
        start_idx = end_idx
        end_idx = start_idx+j+1
        ta_1 = x[0,start_idx:end_idx-1]
        if diag_exp:
            ta_2 = tf.math.exp(x[0,end_idx-1])
        else:
            ta_2 = x[0,end_idx-1]
        ta_2 = tf.reshape(ta_2,shape=(1,))
        ta_3 = tf.zeros(dim-j-1,dtype=tf.float64)
        ta = tf.concat([ta_1,ta_2, ta_3],axis=0)
        ta = tf.reshape(ta,shape=[1,dim])
        tril_mat = tril_mat.write(j,ta)
        
    tril_mat = tril_mat.stack()
    tril_mat = tf.reshape(tril_mat,shape=(dim,dim))        
    return tril_mat
    
# Function to convert  a vectorized parameters to GMM params (alpha, mu, cov)
def vec2param_gmm(theta,ndims,ncomps):
    num_alpha_params = ncomps
    num_mu_params = ncomps*ndims
    num_sig_params = int(ncomps*ndims*(ndims+1)*0.5)

    alpha_param, mu_param, chol_param = tf.split(theta,[num_alpha_params,num_mu_params,num_sig_params])
    alpha = tf.math.softmax(alpha_param)
    mu_vectors = tf.reshape(mu_param, shape=(ncomps,ndims))
       
    chol_mat_array=tf.TensorArray(tf.float64,size=ncomps)
    cov_mat_array=tf.TensorArray(tf.float64,size=ncomps)
    for k in range(ncomps):
        start_idx = tf.cast(k*(num_sig_params/ncomps),tf.int64)
        end_idx = tf.cast((k+1)*(num_sig_params/ncomps),tf.int64)
        chol_mat = vector2Tril(chol_param[start_idx:end_idx])
        chol_mat_array = chol_mat_array.write(k,chol_mat) 
        cov_mat = tf.matmul(chol_mat,tf.transpose(chol_mat))
        cov_mat_array =  cov_mat_array.write(k,cov_mat) 
        
    chol_matrices = chol_mat_array.stack()
    cov_matrices = cov_mat_array.stack()
    return alpha, mu_vectors,chol_matrices,cov_matrices

# Function to convert vectorized parameters to the marginal GMM params (all marginal GMMs 
# are assumed to have same number of components)
def vec2param_marg(theta,n_dims):
    n_total = theta.shape.as_list()[0]
    n_split = int(n_total/3)
    n_comps = int(n_split/n_dims)
    alpha_param, mu_param, var_param = tf.split(theta,[n_split,n_split,n_split])
    alpha = tf.reshape(alpha_param,shape=(n_comps,n_dims))
    alpha = tf.math.softmax(alpha,axis=0)
    mu_vectors = tf.reshape(mu_param,shape=(n_comps,n_dims))
    var_vectors = tf.reshape(var_param,shape=(n_comps,n_dims))
    var_vectors = tf.math.exp(var_vectors)
    return alpha,mu_vectors,var_vectors

# Function that instantiates a GMC distribution given vectorized parameters 
def vec2gmc_dist(theta,ndims,ncomps):
    alpha, mu_vectors,chol_matrices,_ = vec2param_gmm(theta,ndims,ncomps)
    # Defining the base GMM distribution
    gmm_base = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                                components_distribution=tfd.MultivariateNormalTriL(loc=mu_vectors,scale_tril=chol_matrices))
    # Defining the GMC bijector
    gmc_bijector = GMC_bijector(ndims,ncomps,[alpha,mu_vectors,chol_matrices])
    # Defining the GMC distribution
    gmc_distribution = tfp.distributions.TransformedDistribution(gmm_base, gmc_bijector)
    return gmc_distribution

# Function that instantiates a GMCM distribution given vectorized parameters 
def vec2gmcm_dist(gmc_params,marg_params,ndims,ncomps):
    # num_params_all = theta.shape.as_list()[0]
    # num_gmc_params = theta_gmc.shape.as_list()[0] 
    # marg_params, gmc_params =  tf.split(theta,[num_marg_params,num_gmc_params])
    # Specifying the GMC bijector
    alpha, mu_vectors,chol_matrices,_ = vec2param_gmm(gmc_params,ndims,ncomps)
    gmc_bijector = GMC_bijector(ndims,ncomps,[alpha,mu_vectors,chol_matrices])
    # Specifying the marginal icdf bijector
    marg_alpha_vectors, marg_mu_vectors,marg_var_vectors = vec2param_marg(marg_params,ndims)
    gmm_icdf_bijector = mv_GMM_icdf_bijector([marg_alpha_vectors,marg_mu_vectors,marg_var_vectors])
    # Chaining the two bijectors
    gmcm_bijector = tfb.Chain([gmm_icdf_bijector,gmc_bijector])
    # Now, specifying the base GMM distribution
    gmm_base = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                                components_distribution=tfd.MultivariateNormalTriL(loc=mu_vectors,scale_tril=chol_matrices))
   
    # Finally Defining the gmcm distribution
    gmcm_distribution = tfp.distributions.TransformedDistribution(gmm_base, gmcm_bijector)
    return gmcm_distribution

# Function that instantiates a GMCM distribution given vectorized parameters 
def param2gmcm_dist(gmc_params,marg_gmm_params,bounds,ndims,ncomps):
    # Specifying the GMC bijector
    alpha, mu_vectors,chol_matrices,_ = vec2param_gmm(gmc_params,ndims,ncomps)
    gmc_bijector = GMC_bijector(ndims,ncomps,[alpha,mu_vectors,chol_matrices])
    # Specifying the marginal icdf bijector
    marg_alpha_vectors, marg_mu_vectors,marg_var_vectors = vec2param_marg(marg_gmm_params,ndims)
    gmm_icdf_bijector = mv_GMM_icdf_bijector([marg_alpha_vectors,marg_mu_vectors,marg_var_vectors])
    # Specifying the sigmoid transform bijector
    lb,ub = bounds
    sigmoid_bijector = tfb.Sigmoid(low=lb,high=ub)
    # Chaining the three bijectors
    gmcm_bijector = tfb.Chain([sigmoid_bijector, gmm_icdf_bijector, gmc_bijector])
    # Now, specifying the base GMM distribution
    gmm_base = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                                components_distribution=tfd.MultivariateNormalTriL(loc=mu_vectors,scale_tril=chol_matrices))
   
    # Finally Defining the gmcm distribution
    gmcm_distribution = tfp.distributions.TransformedDistribution(gmm_base, gmcm_bijector)
    return gmcm_distribution

# Learning the marginal distribution first as a mixture of Gaussians distribution and synthesizing a vector of all parameters
def fitMargGMMs(obs_mat,bounds,ncomps=10):
    # Learning the marginal distribution first as a mixture of Gaussians distribution and synthesizing a vector of all parameters
    nsamps,ndims = np.shape(obs_mat)
    # log transforming first
    lb,ub = bounds
    obs_mat = tf.math.log((obs_mat-lb)/(ub-obs_mat))
    # learning the GMM distribution on log-transformed data
    u_mat = np.zeros_like(obs_mat)
    marg_alpha_vectors = tf.TensorArray(tf.float64,size=ndims)
    marg_mu_vectors = tf.TensorArray(tf.float64,size=ndims)
    marg_vars = tf.TensorArray(tf.float64,size=ndims)
    for j in range(ndims): 
        gmm = mixture.GaussianMixture(n_components=ncomps, covariance_type='full', max_iter=300, n_init=5, 
                                init_params='kmeans')
        gmm.fit(obs_mat[:,j].numpy().reshape(-1,1))
        curr_alpha = gmm.weights_.tolist()
        curr_mu = gmm.means_.flatten().tolist()
        curr_var = gmm.covariances_.flatten().tolist()
        curr_u = gmm_cdf(tf.cast(obs_mat[:,j],tf.float32),curr_alpha,curr_mu,curr_var)
        u_mat[:,j] = curr_u.numpy()
        marg_alpha_vectors = marg_alpha_vectors.write(j,gmm.weights_.flatten())
        marg_mu_vectors = marg_mu_vectors.write(j,gmm.means_.flatten())
        marg_vars = marg_vars.write(j,gmm.covariances_.flatten())

    marg_alpha_vectors = tf.math.log(tf.transpose(marg_alpha_vectors.stack()))
    marg_mu_vectors = tf.transpose(marg_mu_vectors.stack())
    marg_log_vars = tf.math.log(tf.transpose(marg_vars.stack()))

    marg_params = tf.concat([tf.reshape(marg_alpha_vectors,[-1]),tf.reshape(marg_mu_vectors,[-1]), tf.reshape(marg_log_vars,[-1])],axis=0)

    u_mat = tf.constant(u_mat,dtype=tf.float64)
    return u_mat, marg_params

# # Learning the marginal distribution first as a mixture of Gaussians distribution and synthesizing a vector of all parameters
# def fitMargGMMs(obs_mat,ncomps=10):
#     # Learning the marginal distribution first as a mixture of Gaussians distribution and synthesizing a vector of all parameters
#     nsamps,ndims = np.shape(obs_mat)
#     u_mat = np.zeros_like(obs_mat)
#     marg_alpha_vectors = tf.TensorArray(tf.float64,size=ndims)
#     marg_mu_vectors = tf.TensorArray(tf.float64,size=ndims)
#     marg_vars = tf.TensorArray(tf.float64,size=ndims)
#     for j in range(ndims): 
#         gmm = mixture.GaussianMixture(n_components=ncomps, covariance_type='full', max_iter=300, n_init=5, 
#                                 init_params='kmeans')
#         gmm.fit(obs_mat[:,j].numpy().reshape(-1,1))
#         curr_alpha = gmm.weights_.tolist()
#         curr_mu = gmm.means_.flatten().tolist()
#         curr_var = gmm.covariances_.flatten().tolist()
#         curr_u = gmm_cdf(tf.cast(obs_mat[:,j],tf.float32),curr_alpha,curr_mu,curr_var)
#         u_mat[:,j] = curr_u.numpy()
#         marg_alpha_vectors = marg_alpha_vectors.write(j,gmm.weights_.flatten())
#         marg_mu_vectors = marg_mu_vectors.write(j,gmm.means_.flatten())
#         marg_vars = marg_vars.write(j,gmm.covariances_.flatten())

#     marg_alpha_vectors = tf.math.log(tf.transpose(marg_alpha_vectors.stack()))
#     marg_mu_vectors = tf.transpose(marg_mu_vectors.stack())
#     marg_log_vars = tf.math.log(tf.transpose(marg_vars.stack()))

#     marg_params = tf.concat([tf.reshape(marg_alpha_vectors,[-1]),tf.reshape(marg_mu_vectors,[-1]), tf.reshape(marg_log_vars,[-1])],axis=0)

#     u_mat = tf.constant(u_mat,dtype=tf.float64)
#     return u_mat, marg_params

# Function to initialize GMC params
def initGMCParams(obs_mat,ndims,ncomps,method='gmm'):    
    if method == 'gmm':
        # now fitting the multivariate gaussian mixture
        gmm_mvn = mixture.GaussianMixture(n_components=ncomps,covariance_type='full',max_iter=500,n_init=5,verbose=0)
        gmm_mvn.fit(obs_mat)
        
        alpha = gmm_mvn.weights_
        mu_vectors = gmm_mvn.means_
        cov_matrices = gmm_mvn.covariances_
    elif method == 'random':
        alpha = np.random.uniform(size=(ncomps,))
        alpha = alpha/np.sum(alpha)
        mu_vectors = np.random.normal(size=(ncomps,ndims))
        cov_matrices = np.tile(np.eye(ndims),(ncomps,1,1))
        
    # scale-shift transform to standardize the GMM
    alpha,mu_vectors,cov_matrices = standardizeGMM(alpha,mu_vectors,cov_matrices)
    alpha_params = np.log(alpha)
    mu_params = mu_vectors.flatten()
    chol_params = np.array([])
    tril_idx = np.tril_indices(ndims)
    for k in range(ncomps):
        chol_mat = np.linalg.cholesky(cov_matrices[k])
        for j in range(ndims):
            chol_mat[j,j] = np.log(chol_mat[j,j])
        chol_params = np.concatenate((chol_params,chol_mat[tril_idx[0],tril_idx[1]]),axis=0)

    gmc_params = np.concatenate((alpha_params,mu_params,chol_params),axis=0)
    return gmc_params

def standardizeGMM(alpha,mu_vectors,cov_matrices):
    ncomps,ndims = mu_vectors.shape
    var_vectors = np.zeros((ncomps,ndims),dtype='float64')
    for j in range(ndims):
        var_vectors[:,j] = cov_matrices[:,j,j]
    weighted_mean = np.matmul(mu_vectors.T,alpha)
    new_mu = (mu_vectors-weighted_mean)
    weighted_variance = np.matmul((var_vectors + new_mu**2).T,alpha)

    scaling_matrix = np.diag(weighted_variance**(-0.5))
    new_mu = np.matmul(new_mu,scaling_matrix)
    new_cov = np.zeros((ncomps,ndims,ndims),dtype='float64')
    for k in range(ncomps):
        new_cov[k,:,:] = np.matmul(scaling_matrix,np.matmul(cov_matrices[k,:,:],scaling_matrix))
    return alpha, new_mu, new_cov
    
    # ncomps,ndims = mu_vectors.shape.as_list()
    # var_vectors = tf.TensorArray(tf.float64,size=ndims)
    # for j in range(ndims):
    #     var_vectors = var_vectors.write(j,cov_matrices[:,j,j])
    # var_vectors = tf.transpose(var_vectors.stack())

    # weighted_mean = tf.linalg.matvec(tf.transpose(mu_vectors),alpha)
    # new_mu = (mu_vectors-weighted_mean)
    # weighted_variance = tf.linalg.matvec(tf.transpose(var_vectors + tf.pow(new_mu,2)),alpha)

    # scaling_matrix = tf.linalg.diag(tf.pow(weighted_variance,-0.5))
    # new_mu = tf.linalg.matmul(new_mu,scaling_matrix)
    # new_cov = tf.TensorArray(tf.float64,size=ncomps)
    # for k in range(ncomps):
    #     new_cov = new_cov.write(k,tf.linalg.matmul(scaling_matrix,tf.linalg.matmul(cov_matrices[k,:,:],scaling_matrix)))
    # new_cov = new_cov.stack()
    # return alpha, new_mu, new_cov

# GMC Params prior
def identifiabilityTransfrom(theta,ndims,ncomps):
    alpha, mu_vectors,chol_matrices,cov_matrices = vec2param_gmm(theta,ndims,ncomps)
    var_vectors = tf.TensorArray(tf.float64,size=ndims)
    for j in range(ndims):
        var_vectors = var_vectors.write(j,cov_matrices[:,j,j])
    var_vectors = tf.transpose(var_vectors.stack())


    alpha = tf.reshape(alpha,shape=(ncomps,1))
    vec1=tf.reduce_sum(alpha*mu_vectors,axis=0)
    vec2=tf.reduce_sum(alpha*(var_vectors + tf.pow(mu_vectors,2)),axis=0) - tf.ones(ndims,dtype=tf.float64)
    
    return tf.concat([vec1,vec2],axis=0)
    
def fitMargNonParam(obs_mat):
    x1_array = obs_mat.numpy()
    x1_array_sorted = np.zeros_like(x1_array)
    u1_array = np.zeros_like(x1_array)
    x2_array = np.zeros_like(x1_array)
    u2_array = np.zeros_like(x1_array)
    nsamps,ndims = x1_array.shape
    for j in range(ndims):
        curr_obs = x1_array[:,j] + np.random.normal(0,1E-6,nsamps) # adding a small noise to maintain unique ness of samples
        ranks = np.empty_like(curr_obs)
        ranks[np.argsort(curr_obs)] = np.arange(nsamps)
        x1_array_sorted[:,j] = np.sort(curr_obs)
        u1_array[:,j] = ranks/(nsamps-1)
        x2_array[:,j] = np.linspace(np.min(curr_obs),np.max(curr_obs),nsamps)
        u2_array[:,j] = scipy_interp1d(x2_array[:,j],curr_obs,u1_array[:,j],method='linear')
    # specifying non-parametric marginals as a dict
    marg_params = {'u_range': [np.min(u1_array,axis=0), np.max(u1_array,axis=0)], \
                    'x_values': x1_array_sorted,\
                    'x_range': [np.min(x2_array,axis=0),np.max(x2_array,axis=0)], \
                    'u_values': u2_array}    
    return u1_array, marg_params
    

    
