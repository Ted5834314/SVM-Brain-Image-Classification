import scipy.io
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import os
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
mat = scipy.io.loadmat('label.mat')
np.random.seed(31415)
random.seed(31415)
y=np.array(pd.DataFrame(mat['label']))
y=y.flatten()
images=nib.load('sub-01_ses-test_task-fingerfootlips_bold.nii')
data= images.get_fdata()
data=np.moveaxis(data, -1, 0)
X=data.reshape(184,64*64*30)
mean_feature=np.mean(X, axis=0)
#this function generates a mask with threshold=the pth percentile of the mean_feature where p=percent

def apply_mask(percent,X,mean_feature):
  thresh=np.percentile(mean_feature, percent, axis=0)
  mask=[]
  for i in range(mean_feature.size):
    if mean_feature[i]>thresh:
      mask.append(i)
  newX=[]
  for i in range(184):
    newX.append(X[i][mask])
  return np.array(newX)


#The tuning process is as follows:
#(1)First fix regularization parameter and find a reasonable k value for k-fold cross validation
#(2)Then fix some reasonable parameters outside SVM(like threshold) and use them to do grid search and find an optimal parameter set for SVM

#Some comments are explaining the printed outcomes, please run the code first then read the corresponding comments
#I am using stratified cross validation which splits the dataset according to the proportion of labels
#First tune the regularization parameter
#Note that I am using the grid search syntax even when building individual models, although the
#size of the grid is just 1
print("TEST WITHOUT PCA")
print("")
print("score for reg_param=[0.1,1,10,100,1000,10000,50000,100000]:")
newX=apply_mask(20,X,mean_feature)

reg_perf_lst=[]
skfold = StratifiedKFold(n_splits=8, shuffle=True,random_state=31415)# using stratified cross validation
for reg_param in [0.1,1,10,100,1000,10000,50000,100000]:
  
  param_grid = {'C':[reg_param],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  reg_perf_lst.append(grid.best_score_)
print(reg_perf_lst)
# from above, the result stablize for reg_param=10000 we can set reg_param=10000

#now look at different number of folds
print("score for k=[4,8,12,16,20,24,28]:")
split_perf_lst=[]
for split in [4,8,12,16,20,24,28]:
  skfold = StratifiedKFold(n_splits=split, shuffle=True,random_state=31415)# using stratified cross validation
  param_grid = {'C':[10000],'random_state':[31415]}
  grid = GridSearchCV(SVC(random_state=31415), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  split_perf_lst.append(grid.best_score_)
print(split_perf_lst)

#it can be observed that after the cv number is above 12, the cross validation accuracy is stablized
#when the number of fold is too small it may be under fitting
#so we can just set the cv number to 12 and do grid search

#now observe the influence of threshold
print("score for p=[0,10,20,30,40,50,60,70,80,90] where brain mask threshold=pth percentile of mean image:")
thresh_perf_lst=[]
skfold = StratifiedKFold(n_splits=12, shuffle=True,random_state=31415)# using stratified cross validation
for thresh in [0,10,20,30,40,50,60,70,80,90]:
  newX=apply_mask(thresh,X,mean_feature)

  param_grid = {'C':[10000],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  thresh_perf_lst.append(grid.best_score_)
print(thresh_perf_lst)
#not very influential, but seems to stablize after 60 and reaches peak at 90 so just set it to 90
#now we can do grid search by fixing these default parameters
print("grid search result of optimal tuning parameters")
newX=apply_mask(90,X,mean_feature)
skfold = StratifiedKFold(n_splits=12, shuffle=True,random_state=31415)
param_grid = {'kernel':  ['linear','poly','rbf','sigmoid'],'C':[10000],"gamma":['scale','auto'],'class_weight':['balanced',None],'random_state':[31415]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
grid.fit(newX, y)
print("best cv accuracy: {:.3f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("")
print("")

#now try PCA
print("TEST WITH PCA")
newX=apply_mask(40,X,mean_feature)
scaler = StandardScaler()
scaler.fit(newX)
norm_newX=scaler.transform(newX)
pca = PCA(n_components=50)
X_pca=pca.fit_transform(norm_newX) #a PCA example
#now see if PCA will change the pattern of regularization term
print("score for reg_param=[0.1,1,10,100,1000,10000,50000,100000]:")
reg_perf_lst=[]
skfold = StratifiedKFold(n_splits=8, shuffle=True,random_state=31415)# using stratified cross validation
for reg_param in [0.1,1,10,100,1000,10000,50000,100000]:

  param_grid = {'C':[reg_param],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  reg_perf_lst.append(grid.best_score_)
print(reg_perf_lst)
#it seems that we should still maintain high regularization term, but much lower than non PCA model.

#same for number of k
split_perf_lst=[]
print("score for k=[4,8,12,16,20,24,28]:")
for split in [4,8,12,16,20,24,28]:
  skfold = StratifiedKFold(n_splits=split, shuffle=True,random_state=31415)
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  split_perf_lst.append(grid.best_score_)
print(split_perf_lst)
#it seems the performance is similar for different k>=12, just set it to 12
#this result is not stable with different seeds

#pick an optimal number of PCA components by mean cross-validation accuracy over different regularization parameters
np.random.seed(31415)
random.seed(31415)
opt_score=-1
opt_feature_num=-1
score_lst=[]
skfold = StratifiedKFold(n_splits=12, shuffle=True,random_state=31415)# using stratified cross validation
for feature_num in range(10,180,10):
  pca = PCA(n_components=feature_num)
  X_pca=pca.fit_transform(norm_newX)
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  score=(grid.best_score_)
  score_lst.append(score)
print("average score for number of pca components=10-170:")
print(score_lst)
#It is can be seen that too high or too small num_components would hurt performance
#Note that this parameter has high variability, so I may not choose the optimal one here
#the optimal range is between 20 to 90 and the optimal parameter may vary for different seeds
#60 or 70 is generally a good choice so we set it to be 70
np.random.seed(31415)
random.seed(31415)
thresh_perf_lst=[]
skfold = StratifiedKFold(n_splits=12, shuffle=True,random_state=31415)
print("score for p=[0,10,20,30,40,50,60,70,80,90] where brain mask threshold=pth percentile of mean image:")
for thresh in [0,10,20,30,40,50,60,70,80,90]:
  newX=apply_mask(thresh,X,mean_feature)
  scaler = StandardScaler()
  scaler.fit(newX)
  norm_newX=scaler.transform(newX)
  pca = PCA(n_components=70)
  X_pca=pca.fit_transform(norm_newX)
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  thresh_perf_lst.append(grid.best_score_)
print(thresh_perf_lst)
#it can be seen that from 70 the performance becomes higher
#again this result is not stable for different seeds
#But for this seed it is quite stable.
#Also 60 or 70 is a good choice for other seeds generally, so I pick 70

np.random.seed(31415)
random.seed(31415)
print("grid search result of optimal tuning parameters")
newX=apply_mask(70,X,mean_feature)
scaler = StandardScaler()
scaler.fit(newX)
norm_newX=scaler.transform(newX)
pca = PCA(n_components=70)
X_pca=pca.fit_transform(norm_newX)
skfold = StratifiedKFold(n_splits=12, shuffle=True,random_state=31415)
param_grid = {'kernel':  ['linear','poly','rbf','sigmoid'],'C':[10],"gamma":['scale','auto'],'class_weight':['balanced',None],'random_state':[31415]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
grid.fit(X_pca, y)
print("best cv accuracy: {:.3f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
param_pca=grid.best_params_
print("")
print("")

#RETEST
images=nib.load('sub-01_ses-retest_task-fingerfootlips_bold.nii')
np.random.seed(31415)
random.seed(31415)
data= images.get_fdata()
data=np.moveaxis(data, -1, 0)
X=data.reshape(184,64*64*30)
mean_feature=np.mean(X, axis=0)
#Again, first tune regularization term
print("RETEST WITHOUT PCA")
print("")
print("score for reg_param=[0.1,1,10,100,1000,10000,50000,100000]:")
newX=apply_mask(20,X,mean_feature)

reg_perf_lst=[]
skfold = StratifiedKFold(n_splits=8, shuffle=True,random_state=31415)# using stratified cross validation
for reg_param in [0.1,1,10,100,1000,10000,50000,100000]:
  
  param_grid = {'C':[reg_param]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  reg_perf_lst.append(grid.best_score_)
print(reg_perf_lst)
#from above, the result stablize for reg_param=10000 we can set reg_param=10000
#now look at different number of folds

print("score for k=[4,8,12,16,20,24,28]:")
split_perf_lst=[]
for split in [4,8,12,16,20,24,28]:
  skfold = StratifiedKFold(n_splits=split, shuffle=True,random_state=31415)# using stratified cross validation
  param_grid = {'C':[10000],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  split_perf_lst.append(grid.best_score_)
print(split_perf_lst)
#it can be observed that after the cv number is above 20, the cross validation accuracy stablizes

#now observe the influence of threshold
print("score for p=[0,10,20,30,40,50,60,70,80,90] where brain mask threshold=pth percentile of mean image:")
np.random.seed(31415)
random.seed(31415)
thresh_perf_lst=[]
skfold = StratifiedKFold(n_splits=20, shuffle=True,random_state=31415)# using stratified cross validation
for thresh in [0,10,20,30,40,50,60,70,80,90]:
  newX=apply_mask(thresh,X,mean_feature)
  param_grid = {'C':[10000],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(newX, y)
  thresh_perf_lst.append(grid.best_score_)
print(thresh_perf_lst)
#it seems 80 is a good value, so use 80, again, this result may not be the same for other seeds
#now we can do grid search by fixing these default parameters
print("grid search result of optimal tuning parameters")
np.random.seed(31415)
random.seed(31415)
newX=apply_mask(80,X,mean_feature)
skfold = StratifiedKFold(n_splits=20, shuffle=True,random_state=31415)
param_grid = {'kernel':  ['linear','poly','rbf','sigmoid'],'C':[10000],"gamma":['scale','auto'],'class_weight':['balanced',None],'random_state':[31415]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
grid.fit(newX, y)
print("best cv accuracy: {:.3f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("")
print("")
#now try PCA
np.random.seed(31415)
random.seed(31415)
print("RETEST WITH PCA")
newX=apply_mask(50,X,mean_feature)
scaler = StandardScaler()
scaler.fit(newX)
norm_newX=scaler.transform(newX)
pca = PCA(n_components=60)
X_pca=pca.fit_transform(norm_newX) #a PCA example
#now see if PCA will change the pattern of regularization term
print("score for reg_param=[0.1,1,10,100,1000,10000,50000,100000]:")
reg_perf_lst=[]
skfold = StratifiedKFold(n_splits=8, shuffle=True,random_state=31415)# using stratified cross validation
for reg_param in [0.1,1,10,100,1000,10000,50000,100000]:

  param_grid = {'C':[reg_param],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  reg_perf_lst.append(grid.best_score_)
print(reg_perf_lst)
#again, reg_param=10 is enough

#now find a good choice of number of folds k

split_perf_lst=[]
print("score for k=[4,8,12,16,20,24,28]:")
for split in [4,8,12,16,20,24,28]:
  skfold = StratifiedKFold(n_splits=split, shuffle=True,random_state=31415)
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  split_perf_lst.append(grid.best_score_)
print(split_perf_lst)
#it seems the performance is similar for different k<28. Whcn k=28 the performance is much better but here weset k=20 so it has same number of folds as non-PCA model
#pick an optimal number of PCA components by mean cross-validation accuracy over different regularization parameters
np.random.seed(31415)
random.seed(31415)
opt_score=-1
opt_feature_num=-1
score_lst=[]
skfold = StratifiedKFold(n_splits=20, shuffle=True,random_state=31415)# using stratified cross validation
for feature_num in range(10,180,10):
  pca = PCA(n_components=feature_num)
  X_pca=pca.fit_transform(norm_newX)  
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  score=(grid.best_score_)
  score_lst.append(score)
print("average score for number of pca components=10-170:")
print(score_lst)
#It is can be seen that too high or too small num_components would hurt performance
#the optimal range is between 20 to 90 and the optimal parameter may vary for different seeds
#80 is generally a good choice so we set it to be 80, note that here we don't set it to be the optimal value
#Because randomness still exists and 80 is good generally


np.random.seed(31415)
random.seed(31415)
thresh_perf_lst=[]
skfold = StratifiedKFold(n_splits=20, shuffle=True,random_state=31415)
print("score for p=[0,10,20,30,40,50,60,70,80,90] where brain mask threshold=pth percentile of mean image:")
for thresh in [0,10,20,30,40,50,60,70,80,90]:
  newX=apply_mask(thresh,X,mean_feature)
  scaler = StandardScaler()
  scaler.fit(newX)
  norm_newX=scaler.transform(newX)
  pca = PCA(n_components=80)
  X_pca=pca.fit_transform(norm_newX)
  param_grid = {'C':[10],'random_state':[31415]}
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
  grid.fit(X_pca, y)
  thresh_perf_lst.append(grid.best_score_)
print(thresh_perf_lst)
#it can be seen that from 60 the performance becomes higher, and the accuracy peaked at 80 so pick thresh=80
#again this result is quite unstable, some times there would be big fluctuation for different seeds
#but generally 80 is a good choice, and threshold is more stable than # of PCA components

np.random.seed(31415)
random.seed(31415)
print("grid search result of optimal tuning parameters")
newX=apply_mask(80,X,mean_feature)
scaler = StandardScaler()
scaler.fit(newX)
norm_newX=scaler.transform(newX)
pca = PCA(n_components=80)
X_pca=pca.fit_transform(norm_newX)
skfold = StratifiedKFold(n_splits=20, shuffle=True,random_state=31415)
param_grid = {'kernel':  ['linear','poly','rbf','sigmoid'],'C':[10],"gamma":['scale','auto'],'class_weight':['balanced',None],'random_state':[31415]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=skfold)
grid.fit(X_pca, y)
print("best cv accuracy: {:.3f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
