#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import utils as utl


# In[20]:


# Test Bijectors
class GMC_bijector(tfb.Bijector):
    def __init__(self,ndims,ncomps,inputs,forward_min_event_ndims=1, validate_args: bool = False,name="gmc"):
        super(GMC_bijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        self.ndims = ndims
        self.ncomps = ncomps
        self.alpha = inputs[0]
        self.mu_vectors = inputs[1] 
        self.chol_matrices = inputs[2]
        cov_mat_array=tf.TensorArray(tf.float64,size=ncomps)
        for k in range(self.ncomps):
            cov_mat = tf.matmul(self.chol_matrices[k],tf.transpose(self.chol_matrices[k]))
            cov_mat_array =  cov_mat_array.write(k,cov_mat)         
        self.cov_matrices = cov_mat_array.stack()
    
    def _forward(self, x_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):
            u_cur = utl.gmm_cdf(x_mat[:,j],self.alpha,self.mu_vectors[:,j],self.cov_matrices[:,j,j])
            temp_array = temp_array.write(j,u_cur)
        u_mat = tf.transpose(temp_array.stack())
        return u_mat

    
    def _inverse(self, u_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):
            x_cur = utl.gmm_icdf(u_mat[:,j],self.alpha,self.mu_vectors[:,j],self.cov_matrices[:,j,j])
            temp_array = temp_array.write(j,x_cur)
        x_mat = tf.transpose(temp_array.stack())
        return x_mat

    
    def _forward_log_det_jacobian(self,x_mat):
        nobs = x_mat.get_shape().as_list()[0]
        forward_log_det_J = tf.zeros(nobs,dtype=tf.float64)
        for j in range(self.ndims):
            forward_log_det_J += utl.gmm_lpdf(x_mat[:,j],self.alpha,self.mu_vectors[:,j],self.cov_matrices[:,j,j])
        return forward_log_det_J
        
    
    def _inverse_log_det_jacobian(self, u_mat):
        return -self._forward_log_det_jacobian(self._inverse(u_mat))
                                                             
# Test Bijectors
class mv_GMM_icdf_bijector(tfb.Bijector):
    def __init__(self,inputs,forward_min_event_ndims=1, validate_args: bool = False,name="gmm_icdf"):
        super(mv_GMM_icdf_bijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        self.alpha = inputs[0]
        self.mu_vectors = inputs[1]
        self.vars = inputs[2]
        ncomps,ndims = self.mu_vectors.shape.as_list()                                                          
        self.ndims = ndims
        self.ncomps = ncomps
    
    def _forward(self, u_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):            
            x_cur = utl.gmm_icdf_alt(u_mat[:,j],self.alpha[:,j],self.mu_vectors[:,j],self.vars[:,j])
            temp_array = temp_array.write(j,x_cur)
        x_mat = tf.transpose(temp_array.stack())
        return x_mat

    
    def _inverse(self, x_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):
            u_cur = utl.gmm_cdf(x_mat[:,j],self.alpha[:,j],self.mu_vectors[:,j],self.vars[:,j])
            temp_array = temp_array.write(j,u_cur)
        u_mat = tf.transpose(temp_array.stack())
        return u_mat

    
    def _inverse_log_det_jacobian(self,x_mat):
        nobs = x_mat.get_shape().as_list()[0]
        inverse_log_det_J = tf.zeros(nobs,dtype=tf.float64)
        for j in range(self.ndims):
            inverse_log_det_J += utl.gmm_lpdf(x_mat[:,j],self.alpha[:,j],self.mu_vectors[:,j],self.vars[:,j])
        return inverse_log_det_J
        
    
    def _forward_log_det_jacobian(self, u_mat):
        return -self._inverse_log_det_jacobian(self._inverse(u_mat))
    
    # Test Bijectors
class nonParam_icdf_bijector(tfb.Bijector):
    def __init__(self,inputs,forward_min_event_ndims=1, validate_args: bool = False,name="nonParam_icdf_bijector"):
        super(nonParam_icdf_bijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        self.u_range = inputs[0]
        self.x_range = inputs[1]
        self.u_vals = inputs[2]
        self.x_vals = inputs[3]
        self.ndims = self.u_range[0].size
        self.ngrid = np.size(self.u_vals,0)
    
    def _forward(self, u_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):            
            u_min = self.u_range[0][j]
            u_max = self.u_range[1][j]
            x_cur = tfp.math.interp_regular_1d_grid(u_mat[:,j],u_min,u_max,self.x_vals[:,j])
            temp_array = temp_array.write(j,x_cur)
        x_mat = tf.transpose(temp_array.stack())
        return x_mat

    
    def _inverse(self, x_mat):
        temp_array = tf.TensorArray(tf.float64,size=self.ndims)
        for j in range(self.ndims):
            x_min = self.x_range[0][j]
            x_max = self.x_range[1][j]
            u_cur = tfp.math.interp_regular_1d_grid(x_mat[:,j],x_min,x_max,self.u_vals[:,j])
            temp_array = temp_array.write(j,u_cur)
        u_mat = tf.transpose(temp_array.stack())
        return u_mat

    
    def _inverse_log_det_jacobian(self,x_mat):
        nobs = x_mat.get_shape().as_list()[0]
        inverse_log_det_J = tf.zeros(nobs,dtype=tf.float64)
        for j in range(self.ndims):
            x_min = self.x_range[0][j]
            x_max = self.x_range[1][j]
            delta = (x_max-x_min)/(self.ngrid-1)
            a = tf.constant(np.linspace(x_min,x_max,self.ngrid).astype('float64'),shape=(self.ngrid,1))
            b = x_mat[:,j]
            c = tf.transpose(tf.reshape(tf.repeat(b,repeats=[self.ngrid]*nobs),shape=(nobs,self.ngrid)))
            d = a-c
            e = tf.where(d<0, tf.zeros_like(d), d)
            f=tf.where(e>delta, tf.zeros_like(e), e)
            idx = tf.where(tf.transpose(f)!=0)[:,1]
            # jac = (tf.gather(self.u_vals[:,j],idx) -tf.gather(self.u_vals[:,j],idx-1))/delta
            jac = (tf.gather(self.u_vals[:,j],idx-2)-8*tf.gather(self.u_vals[:,j],idx-1) \
                    +8*tf.gather(self.u_vals[:,j],idx)-tf.gather(self.u_vals[:,j],idx+1))/(12*delta)
            inverse_log_det_J += tf.math.log(jac)
        return inverse_log_det_J  
        
    
    def _forward_log_det_jacobian(self, u_mat):
        return -self._inverse_log_det_jacobian(self._inverse(u_mat))

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