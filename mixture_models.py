import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
import time
import utils_v1 as utl
from custom_bijectors_v1 import Marginal_transform, GMC_bijector

# Defining GMC class
class GMC:
    def __init__(self, n_dims, n_comps, param_vec):        
        self.ndims = n_dims
        self.ncomps = n_comps   
        self.total_trainable_params = int(n_comps*(1+n_dims+0.5*n_dims*(n_dims+1)))
        self.params = param_vec
        assert tf.size(param_vec) == self.total_trainable_params, 'the supplied parameter vector is not commensurate with the n_dims, and n_comps'
        
    @property
    def distribution(self):
        # transforming vector in to parameters
        logits,mus,covs,chols = utl.vec2gmm_params(self.ndims,self.ncomps,self.params)
        # Instantiating the bijector
        gmc_bijector = GMC_bijector(self.ndims, self.ncomps, [logits, mus, tf.linalg.diag_part(covs)])
        # Specifying the base distribution
        base_dist = tfd.MixtureSameFamily(tfd.Categorical(logits=logits),
                                          tfd.MultivariateNormalTriL(loc=mus,scale_tril=chols))
        # Instnatiating the gmc distribution as a transformed distribtution
        gmc_dist = tfd.TransformedDistribution(distribution=base_dist,bijector=gmc_bijector)    
        return gmc_dist   
    
    @property
    def identifiability_prior(self):
        # transforming vector in to parameters
        logits,mus,covs,_ = utl.vec2gmm_params(self.ndims,self.ncomps,self.params)        
        alphas = tf.math.softmax(logits)
        variances = tf.linalg.diag_part(covs)        
        vec1 = tf.linalg.matvec(tf.transpose(mus),alphas)
        vec2 = tf.linalg.matvec(tf.transpose(variances + mus**2),alphas)
        log_prior_1 = tfd.MultivariateNormalDiag(loc=tf.zeros(self.ndims),scale_diag=1E-1*tf.ones(self.ndims)).log_prob(vec1)
        log_prior_2 = tfd.MultivariateNormalDiag(loc=tf.ones(self.ndims) ,scale_diag=1E-1*tf.ones(self.ndims)).log_prob(vec2)
        return log_prior_1,log_prior_2
    

    
class GMCM:
    def __init__(self, 
                 n_dims, 
                 data_in, 
                 log_transform_data=True, 
                 data_split=[0.7, 0.3], 
                 marginals_list=None, 
                 gmc=None):
        
        self.ndims = n_dims
        self.data_in = data_in
        self.gmc = gmc
        if gmc:
            self.ncomps = gmc.ncomps
            
        # Transforming input data if specified 
        if log_transform_data:
            min_val = np.min(data_in).astype('float32')-1
            self.log_transform = tfb.Chain([tfb.Shift(shift=min_val.astype('float32')),tfb.Exp()])
        else:
            self.log_transform = None
        # transform the input data via the bijector
        transformed_data = self.log_transform.inverse(data_in).numpy() if self.log_transform else data_in
        
        # Splitting the data into Training, Testing and Validation set
        if len(data_split)==2:
            num_trn=round(transformed_data.shape[0]*data_split[0])
            num_tst=round(transformed_data.shape[0]*data_split[1])
            num_vld=0
        elif len(data_split)==3:
            num_trn=round(transformed_data.shape[0]*data_split[0])
            num_tst=round(transformed_data.shape[0]*data_split[1])
            num_vld=round(transformed_data.shape[0]*data_split[2])    
        # splitting the data in training, testing and validation sets
        np.random.shuffle(transformed_data)
        data_trn,data_tst,data_vld,_ = np.split(transformed_data,np.cumsum([num_trn,num_tst,num_vld]))
        # saving different datasets as properties
        self.data_trn = data_trn
        self.data_tst = data_tst
        self.data_vld = data_vld
        
        # Learn the marginal if not pre-specified (as lists) 
        if marginals_list is None:
            print('Learning Marginals')
            ts = time.time()
            marginals_list = self.learn_marginals()
            print(f'Marginals learnt in {np.round(time.time()-ts,2)} s.') 
        self.marg_dists = marginals_list
        self.marg_bijector = Marginal_transform(self.ndims,self.marg_dists)       
        
    @property
    def distribution(self):
        # setting the gmcm distribution as a transformed distribution of gmc_distribution
        gmcm_dist = tfd.TransformedDistribution(distribution=self.gmc.distribution,bijector=self.marg_bijector)
        if self.log_transform:
            gmcm_dist = tfd.TransformedDistribution(distribution=gmcm_dist,bijector=self.log_transform)
        return gmcm_dist
    
    
    def learn_marginals(self):
        # fitting marginal distributions first
        marg_dist_list=[]
        for j in range(self.ndims):
            input_vector = self.data_trn[:,j].reshape(-1,1)
            marg_gmm_obj = utl.GMM_best_fit(input_vector,max_ncomp=10)
            marg_gmm_tfp = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=marg_gmm_obj.weights_.flatten().astype('float32')),
                components_distribution=tfd.Normal(loc=marg_gmm_obj.means_.flatten().astype('float32'),
                scale = marg_gmm_obj.covariances_.flatten().astype('float32')**0.5),)
            
            # creating a dictionary containing necessary information about each marginal distribution
            info_dict={'cdf':marg_gmm_tfp.cdf,
                       'log_pdf':marg_gmm_tfp.log_prob,
                       'lb':tf.reduce_min(input_vector)-3*tfp.stats.stddev(input_vector),
                       'ub':tf.reduce_max(input_vector)+3*tfp.stats.stddev(input_vector)                         
                      }
            
            marg_dist_list.append(info_dict)     
        return marg_dist_list
        
    def init_GMC_params(self,initialization=['random',None]):
        # Initializing the GMC params 
        init_method, seed_val = initialization
        if init_method == 'random':
            if seed_val:
                np.random.seed(seed_val)
            alphas = tf.ones(self.ncomps)/self.ncomps
            mus = tf.constant(np.random.randn(self.ncomps,self.ndims).astype('float32'))
            covs = tf.repeat(tf.expand_dims(tf.eye(self.ndims),0),self.ncomps,axis=0)
        elif init_method == 'gmm':            
            gmm = mixture.GaussianMixture(n_components=self.ncomps,
                                          covariance_type='full',
                                          max_iter=1000,
                                          n_init=5)
            gmm.fit(self.data_trn)
            alphas = gmm.weights_.astype('float32')
            mus = gmm.means_.astype('float32')
            covs = gmm.covariances_.astype('float32')                                                            
        
        # changing the parameters to standardize the resulting gmm
        alphas,mus,covs = utl.standardize_gmm_params(alphas,mus,covs)
        # now initializing trainable parameters
        init_params = tf.Variable(utl.gmm_params2vec(self.ndims,self.ncomps,alphas,mus,covs))
        
        return init_params
    
    
    def fit_dist(self, 
                 n_comps, 
                 optimizer = tf.optimizers.Adam(learning_rate=1E-2), 
                 initialization = ['random',None], 
                 max_iters = 1000, 
                 batch_size = 10, 
                 print_interval=100, 
                 regularize=True, 
                 plot_results = False):
        
        self.ncomps = n_comps
        # getting the marginal CDF values
        u_mat = self.marg_bijector.inverse(self.data_trn)
        # initializing the parameters
        gmc_params = self.init_GMC_params(initialization=initialization)
        # instantiation GMC object
        gmc_obj = GMC(self.ndims,self.ncomps,gmc_params)
        
        # Defining the training step
        @tf.function
        def train_step(u_selected):
            with tf.GradientTape() as tape:
                neg_gmc_ll = -tf.reduce_mean(gmc_obj.distribution.log_prob(u_selected))
                ident_prior = gmc_obj.identifiability_prior
                if regularize:
                    total_cost = neg_gmc_ll - tf.reduce_sum(ident_prior)
                else:
                    total_cost = neg_gmc_ll
                    
            grads = tape.gradient(total_cost, gmc_obj.params)
            if not (tf.reduce_any(tf.math.is_nan(grads)) or tf.reduce_any(tf.math.is_inf(grads))):
                optimizer.apply_gradients(zip([grads], [gmc_obj.params])) #updating the gmc parameters
            return neg_gmc_ll,ident_prior[0],ident_prior[1]

        neg_ll_trn = np.empty(max_iters)  
        neg_ll_trn[:] = np.NaN
        neg_prior_1 = np.empty(max_iters)  
        neg_prior_1[:] = np.NaN
        neg_prior_2 = np.empty(max_iters)  
        neg_prior_2[:] = np.NaN
        np.random.seed(10)
        ts = time.time() # start time
        # Optimization iterations
        for itr in np.arange(max_iters):
            np.random.seed(itr)
            # Executing a training step
            samps_idx = np.random.choice(u_mat.shape[0],batch_size)
            u_selected_trn = tf.gather(u_mat,samps_idx)
            out = train_step(u_selected_trn)
            neg_ll_trn[itr] = out[0].numpy()
            neg_prior_1[itr] = out[1].numpy()
            neg_prior_2[itr] = out[2].numpy()    
            # Printing results every 100 iteration    
            if tf.equal(itr%print_interval,0) or tf.equal(itr,0):
                time_elapsed = np.round(time.time()-ts,1)
                print(f'@ Iter:{itr}, Training error: {neg_ll_trn[itr]}, LogPriors: {np.round(neg_prior_1[itr],2), np.round(neg_prior_2[itr],2)}, Time Elapsed: {time_elapsed} s')    
        
        if plot_results:
            # Plotting results
            plt.plot(neg_ll_trn)
            plt.xlabel('Iteration',fontsize=12)
            plt.ylabel('Neg_logLike',fontsize=12)
            plt.legend(['train'],fontsize=12)
        
        # setting gmc distritbution embedded inside GMCM
        self.gmc = gmc_obj
         
        return neg_ll_trn
    
    def get_marginal(self,dim_list):        
        data_in_new = tf.gather(self.data_in,dim_list,axis=1).numpy()
        logits,mus,covs,_ = vec2gmm_params(self.ndims,self.ncomps,self.gmc.params)
        alphas = tf.math.softmax(logits)
        dim_remove = list(set(list(range(self.ndims)))-set(dim_list))
        mus_new = tf.gather(mus, dim_list, axis=1)
        covs_new = tf.TensorArray(tf.float32,self.ncomps)
        for k in range(self.ncomps):
            temp_mat = covs[k].numpy()
            covs_new = covs_new.write(k,temp_mat[np.ix_(dim_list,dim_list)])
        covs_new = covs_new.stack()
        # getting the gmc object first for the marginal gmcm
        marginal_gmc_params = gmm_params2vec(len(dim_list),self.ncomps,alphas,mus_new,covs_new)
        marg_gmc = GMC(len(dim_list),self.ncomps,marginal_gmc_params)
        # then getting the marginals along the specified dimensions
        marg_list_new = []
        for j in range(self.ndims):
            if j in dim_list:
                marg_list_new.append(self.marg_dists[j])
        # creating the marginal gmcm object
        marg_gmcm_dist = GMCM(len(dim_list), 
                              data_in_new, 
                              log_transform=self.log_transform, 
                              marginals_list=marg_list_new, 
                              gmc=marg_gmc)
        return marg_gmcm_dist   
    
#     def get_conditional(self,obs_dim_list, value_list):
        
#         x_obs = np.array(value_list).reshape(1,-1).astype('float32')
#         unobs_dim_list = list(set(range(self.ndims)) - set(obs_dim_list))
        
#         #Obtaining the marginal distribution for the observed and missing part
#         gmcm_observed = self.get_marginal(obs_dim_list)
#         gmcm_unobserved = self.get_marginal(unobs_dim_list)
        
#         temp_obj = gmcm_observed
#         z_obs = np.copy(x_obs)
#         while hasattr(temp_obj.distribution,'bijector'):
#             z_obs = temp_obj.distribution.bijector.inverse(z_obs).numpy()
#             temp_obj = temp_obj.distribution

#         #Obtaining the conditional mu and Sigma of individual compoents of the missing part given the data of observed part
#         mus_cond = np.zeros((self.ncomps,len(unobs_dim_list))).astype('float32')
#         covs_cond = np.zeros((self.ncomps,len(unobs_dim_list), len(unobs_dim_list))).astype('float32')
#         logits_cond = np.zeros(self.ncomps).astype('float32')
        
#         logits,mus,covs,_ = vec2gmm_params(self.ndims,self.ncomps,self.gmc.params)
#         logits_unobs,mus_unobs,covs_unobs,_ = vec2gmm_params(gmcm_unobserved.ndims,gmcm_unobserved.ncomps,gmcm_unobserved.gmc.params)
#         logits_obs,mus_obs,covs_obs,_ = vec2gmm_params(gmcm_observed.ndims,gmcm_observed.ncomps,gmcm_observed.gmc.params)
        
#         for k in range(self.ncomps):
#             sig_11 = covs_unobs.numpy()[k]
#             sig_22 = covs_obs.numpy()[k]
#             sig_12 = covs.numpy()[k][np.ix_(unobs_dim_list,obs_dim_list)]
#             sig_21 = sig_12.T
#             mu_11 = mus_unobs[k,:]
#             mu_22 = mus_obs[k,:]
            
# #             temp_mat1 = np.concatenate([np.concatenate([sig_11,sig_12],axis=1),np.concatenate([sig_21,sig_22],axis=1)],axis=0)
# #             temp_mat2=covs.numpy()[k]
# #             lll = unobs_dim_list+obs_dim_list
# #             temp_mat2 = temp_mat2[:,lll]
# #             temp_mat2 = temp_mat2[lll,:]
# #             print(temp_mat1-temp_mat2)
            

#             # Getting the conditional mu and Sigma
#             mu_bar = mu_11 + np.matmul(sig_12,  np.linalg.solve(sig_22,z_obs.T)).flatten()
#             sig_bar = sig_11 - np.matmul(sig_12,  np.linalg.solve(sig_22,sig_21))
#             mus_cond[k] = mu_bar
#             covs_cond[k] = (sig_bar+sig_bar.T)/2

#             # Getting the log proability of the components conditioned on the observed data
#             logits_cond[k] = logits_obs[k] + tfd.MultivariateNormalFullCovariance(loc=mus_obs[k],
#                                                                                          covariance_matrix=covs_obs[k]).log_prob(z_obs)
#         #logits to probabilities
#         alphas_cond = tf.math.softmax(logits_cond)
#         # parameter vector of the conditional gmc distribution
#         conditional_gmc_params = gmm_params2vec(len(unobs_dim_list),self.ncomps,alphas_cond,mus_cond,covs_cond)
#         cond_gmc = GMC(len(unobs_dim_list),self.ncomps,conditional_gmc_params)
#         # then getting the marginals along the specified dimensions
#         marg_list_new = []
#         for j in range(self.ndims):
#             if j in unobs_dim_list:
#                 marg_list_new.append(self.marg_dists[j])
#         # creating the conditional gmcm object
#         data_in_new = tf.gather(self.data_in,unobs_dim_list,axis=1).numpy()
#         cond_gmcm_dist = GMCM(len(unobs_dim_list), data_in_new, forward_transform=self.data_transform, marginals_list=marg_list_new, gmc=cond_gmc)
#         return cond_gmcm_dist  
        