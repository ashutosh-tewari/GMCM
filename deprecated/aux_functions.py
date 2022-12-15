# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:24:15 2020

@author: tewari
"""

import numpy as np
import tensorflow as tf
import utils as utl

def gmc_log_pdf(u_mat,theta):
    # getting the dimensions and number of components
    nsamps = np.size(u_mat,0)
    ndims = np.size(u_mat,1)
    npars = theta.get_shape().as_list()[0]
    ncomps = int(npars/(1+ndims+ndims*(ndims+1)/2))

    alpha, mu_vectors, chol_matrices, cov_matrices = utl.vec2param_gmm(theta,ndims,ncomps)
    
    # Getting the inverse values of all data points
    x_inv_all = tf.TensorArray(tf.float64,size=ndims)
    for j in range(ndims):
        x_inv_all = x_inv_all.write(j,utl.gmm_icdf_alt(u_mat[:,j],alpha,mu_vectors[:,j],cov_matrices[:,j,j]))
    x_inv_all = x_inv_all.stack()
    x_inv_all = tf.transpose(x_inv_all)
    
    # log-likelihood first part (of each sample)
    log_q1 = gmc_log_pdf_q1(x_inv_all,alpha,mu_vectors, chol_matrices)
    # log-likelihood second part (of each sample)
    log_q2 =  gmc_log_pdf_q2(x_inv_all,alpha,mu_vectors, chol_matrices)
    # total log-likelihood (point-wise and cumulative)
    gmc_log_likelihood = (log_q1-log_q2)
    gmc_log_likelihood_cumulative =  tf.reduce_sum(gmc_log_likelihood)
            
    return gmc_log_likelihood_cumulative, gmc_log_likelihood

def gmc_log_pdf_q1(x_mat,alpha,mu_vectors,chol_matrices):  
    q1,_ = utl.mv_gmm_lpdf(x_mat,alpha,mu_vectors,chol_matrices) 
    return q1

def gmc_log_pdf_q2(x_mat,alpha,mu_vectors,chol_matrices):
    # getting the number of dimensions and components
    ncomps = alpha.get_shape().as_list()[0]
    nsamps, ndims = x_mat.get_shape().as_list()
    cov_mat_array=tf.TensorArray(tf.float64,size=ncomps)
    for k in range(ncomps):
        cov_mat = tf.matmul(chol_matrices[k,:,:],tf.transpose(chol_matrices[k,:,:]))
        cov_mat_array =  cov_mat_array.write(k,cov_mat) 
  
    cov_matrices = cov_mat_array.stack()    

    # Getting the inverse values
    ll2_temp_array_2 = tf.TensorArray(tf.float64,size=ndims)  
    for j in range(ndims):   
        ll2_temp_array_2 = ll2_temp_array_2.write(j, utl.gmm_lpdf(x_mat[:,j],alpha,mu_vectors[:,j],cov_matrices[:,j,j]) )
    logL_part2_vec = tf.transpose(ll2_temp_array_2.stack())
    # Obtaing the second part of the log-likelihood
    q2 = tf.reduce_sum(logL_part2_vec,axis=1)
    return q2
