# GMCM-TF
Gaussian Mixture Copula Model (GMCM) implementation in TensorFlow-Probability (TFP). GMCM is more expressive alternative to the Gaussian Mixture Models (GMMs), with the same parameterization to encode multimodal dependence structure.

For details, see:
Link-to-the-paper

## Requirements
- python>=3.7
- numpy>=1.19.5
- TensorFlow>=2.5.0
- TensorFlow-Probability>=0.13.0
- scikit-learn>=0.23.2
- pandas>=1.1.3


## Structure
- /mixture_models.py: Implementation of the GMCM class.
- /custom_bijectors.py: Implementation of bijectors needed for GMCM specification
- /utils.py: Various auxiliary functions.
- /exps: Folder with additional code to reproduce experiments in the paper.
- /UCI_data_preproc: Folder with scripts to preprocess data (Copied from the github repo https://github.com/gpapamak/maf)


## Usage
Below are examples to fit GMCM on Iris and Wine datasets. These examples can be found in GMCM_demo.ipynb notebook.

###### Load the dataset from scikit-learn public database
```
from sklearn.datasets import load_wine
data=load_wine().data.astype('float32')
```
Note the conversion to 'float32' is necessary to match with the TensorFlow backend.

###### Initialing GMCM object of dimension ndims
```
gmcm_obj=GMCM(ndims, data_transform=log_transform)
```
The optional argument data_transform is a TFP bijector that transforms the data ```x``` as ```log(x-v)```, where ```v=min(x)-3*std(x)```. This can be helpful to remove heavy-tailed behavior before learning the marginals.

###### Train GMCM (with 2 components)
Fitting the GMC distribution by direct likelihood maximization. The fitting follows the [IFM](https://open.library.ubc.ca/soa/cIRcle/collections/facultyresearchandpublications/52383/items/1.0225985) method, where marginals are learned first followed by GMC distribution. For marginals here, univariate mixture of Gaussians are used, wherein the number of components are ascertained by the BIC.  
```
nll_train,nll_vld,_=gmcm_obj.fit_dist_IFM(data_trn,
                                    n_comps=2,
                                    batch_size=10,
                                    max_iters=5000,
                                    regularize=True,
                                    print_interval=500)
```
The ```gmcm_obj.distribution``` a TFP distribution (TFD) object with access to all the implemented functions e.g. ```tfd.log_prob(), tfd.sample()``` etc.

###### 2-D contour plots of GMCM density after trainining on Iris dataset 
![GMCM_Iris](https://user-images.githubusercontent.com/16651379/214678357-5477e50a-287a-44d6-9b28-0075f2e024d5.png)

###### 2-D contour plots of GMCM density after trainining on Wine dataset 
![GMCM_Wine](https://user-images.githubusercontent.com/16651379/214678434-2e0d38d3-6a48-45e2-8466-1fb6fbb412ad.png)

###### Marginalizing the GMCM distribution
Suppose we want a marginalized a GMCM along the dimensions 1, 2 and 8. This can be done by defining ```dim_list=[1,2,8]``` and invoking the following code
```
gmcm_128 = gmcm_obj.get_marginal(dim_list)
```

## UCI benchmark dataset
Refer to the documentation, *How to get the datasets*?, in https://github.com/gpapamak/maf

