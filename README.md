# GMCM
Gaussian Mixture Copula Model (GMCM) implementation using TensorFlow-Probability (TFP) constructs. GMCM is a more expressive alternative to the Gaussian Mixture Models (GMMs) while having the same parameterization to encode multimodal dependence structure.

For details, see https://proceedings.mlr.press/v202/tewari23a.html 

## Citation
"@InProceedings{pmlr-v202-tewari23a,
  title = 	 {On the Estimation of {G}aussian Mixture Copula Models},
  author =       {Tewari, Ashutosh},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {34090--34104},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
}"

## Requirements
- python>=3.7
- numpy>=1.19.5
- TensorFlow>=2.5.0
- TensorFlow-Probability>=0.13.0
- scikit-learn>=0.23.2
- pandas>=1.1.3


## Structure
- GMCM/mixture_models.py: Implementation of the GMCM class.
- GMCM/custom_bijectors.py: Implementation of bijectors needed for GMCM specification
- GMCM/utils.py: Various auxiliary functions.
- GMCM/exps: Folder with additional code to reproduce experiments in the paper.
- GMCM/UCI_data_preproc: Folder with scripts to preprocess data (Copied from the github repo https://github.com/gpapamak/maf)


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
Fitting the GMC distribution by direct likelihood maximization. The fitting follows the [IFM](https://open.library.ubc.ca/soa/cIRcle/collections/facultyresearchandpublications/52383/items/1.0225985) method, where marginals are learned first followed by the learning of GMC distribution. For marginals here, univariate mixture of Gaussians are used, wherein the number of components are ascertained by the BIC.  
```
nll_train,nll_vld,_=gmcm_obj.fit_dist_IFM(data_trn,
                                    n_comps=2,
                                    batch_size=10,
                                    max_iters=5000,
                                    regularize=True,
                                    print_interval=500)
```
The ```gmcm_obj.distribution``` a TFP distribution (TFD) object with access to all the inbuilt functions e.g. ```tfd.log_prob(), tfd.sample()``` etc.

###### Marginalizing the GMCM distribution
Suppose we want a marginalized a GMCM along the dimensions 1, 2 and 8. This can be done by defining ```dim_list=[1,2,8]``` and invoking the following code
```
gmcm_128 = gmcm_obj.get_marginal(dim_list)
```
Finally, the density contours can be obtained by the marginalizing GMCM along any two specified dimensions and computing the ```log_prob()``` on a 2-d meshgrid.
###### 2-D contour plots of GMCM density after trainining on Iris dataset 
![GMCM_Iris](https://user-images.githubusercontent.com/16651379/214678357-5477e50a-287a-44d6-9b28-0075f2e024d5.png)

###### 2-D contour plots of GMCM density after trainining on Wine dataset 
![GMCM_Wine](https://user-images.githubusercontent.com/16651379/214678434-2e0d38d3-6a48-45e2-8466-1fb6fbb412ad.png)


## UCI benchmark dataset
Refer to the documentation, *How to get the datasets*?, in https://github.com/gpapamak/maf

