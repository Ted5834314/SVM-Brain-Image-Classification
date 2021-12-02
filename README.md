# SVM-Brain-Image-Classification

**First some general comments:

The output of my code will be the test results, corresponding comments are written inside my code
To understand the choice of parameters, please read the comment and corresponding output.

This file will explain the final result and general observations


**For parameters inside SVM, I tuned 4 of them

    1, C: regularization term
    2, kernel: the type of kernel function to be used
    3, class_weight:whether the weight of each classes are the same or based on the distribution of them, balanced means based on the distribution, none means     same.
    4, gamma: the kernel coefficient for 'rbf' 'poly' and 'sigmoid'

The most important one is regularization term, which needs to be high since there are too many features. So I fix C before all other parameters

The other three are tuned using grid search, and the result may not be stable for different seeds and different parameters outside the SVM classifier. So I will not spend too much time explaining these parameters.


**For parameters outside SVM they are:

    1,number of folds k: This is not actually a tuning parameter, but need to set it to some value that stablizes the accuracy, if too low there will be underfitting

    2,threshold for brain mask: I pick threshold based on the percentiles of the mean image, that is the pth percentile would be keeping 1-p/100 proportion of the     image

    3,number of PCA components: too high or too low will hurt the performance, generally the best choice would be between 20 to 90

**In terms of stability of the parameters there are 3 levels:

    1, stable for all seeds: Only the regularization term is stable for all seeds and it is the most influential one
    2, stable for a certain seed: The other parameters inside SVM are stable if we fix a seed
    3, not stable even with seed fixed: threshold for brain mask and number of PCA components are not stable even with the seed fixed. Although this program     would produce the same output, if we run the chunks individually(in Jupyter Notebook) the result would vary a lot especially for PCA experiments. So I am not     necessarily choosing the optimal value for them as shown in the output, instead I choose the ones that generally have high performance across different runs.

**Experiment Result:

    1,The optimal parameter for test without PCA: C=10000, class_weight=balanced, gamma=scale, kernel=linear, threshold=90th percentile of the mean image
     accuracy=0.902
    2,The optimal parameter for test with PCA: C=10, class_weight=balanced, gamma=scale, kernel=linear, threshold=70th percentile of the mean image, number of     PCA components=70
     accuracy=0.930
    3,The optimal parameter for retest without PCA: C=10000, class_weight=balanced, gamma=scale, kernel=poly, threshold=80th percentile of the mean image
     accuracy=0.847
    4,The optimal parameter for retest with PCA: C=10, class_weight=balanced, gamma=scale, kernel=linear, threshold=80th percentile of the mean image, number     of PCA components=80
     accuracy=0.889
From the experiment result we can see that the result of retest is worse than test, maybe because the test images have higher quality. Also we can see our thresholds are high for all the experiments, and the number of PCA components is low(compared with the total number of features) this might to be caused by overfitting, since there are too many features and if we use most of them, there is likely to be overfitting. The only model that uses a polynomial kernel is for retest without PCA, class_weight and gamma are pretty stable for all the 4 models. However this is only for this seed and if we choose other seeds the result may vary.


**Comparison Between PCA and non-PCA:

It is clear that PCA has higher accuracy than non-PCA, which may because of overfitting just as I mentioned. This is illustrated by the low regularization term of PCA, since the result stablizes when C=10, for non-PCA models, however, the result stablizes after C>=10000. In addition, we can pick a slightly lower threshold(which is illustrated by test without PCA and test with PCA) to increase our ROI for PCA, because we are not overfitting as much as non-PCA model. Also, PCA is trained much faster than non-PCA ones because of the lower dimensionality of features. So PCA is better in terms of both performance and efficiency. 

**Limitations:

Some parameters are not stable, by doing grid search we are only finding the oprimal parameter for this specific seed. To find out the real optimal ones, we may need to take votes from different seeds. And if for different seeds there doesn't exist a majority choice of tuning parameters, then the optimal parameter might not exist. In addition, the parameters out side SVM is not tuned with grid search but individually, there might be interaction between these parameters and parameters inside SVM, so they might not be the real optimal choice. To take these interaction into consideration, we may consider doing a global grid search for all parameters, or we can consider tuning the parameters out side SVM again after doing grid search, and repeat the tuning process until the parameters converge.
