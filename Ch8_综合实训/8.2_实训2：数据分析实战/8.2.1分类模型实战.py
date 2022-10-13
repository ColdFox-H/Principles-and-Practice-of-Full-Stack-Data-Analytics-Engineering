#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入类库
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')
import random


# In[2]:


# 导入数据
filename='sonar.all-data.csv'
dataset=read_csv(filename,header=None)


# In[3]:


# 数据维度
print(dataset.shape)


# In[4]:


# 查看数据类型
set_option('display.max_rows',500)
print(dataset.dtypes)


# In[5]:


# 查看最初的20条记录
set_option('display.width',100)
print(dataset.head(20))


# In[6]:


# 描述性统计信息
set_option('precision',3)
print(dataset.describe())


# In[7]:


# 数据的类型分布
print(dataset.groupby(60).size())


# In[8]:


# 直方图
dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
plt.show()


# In[9]:


# 密度图
dataset.plot(kind='density',subplots=True,layout=(8,8),sharex=False,legend=False,fontsize=1)
plt.show()


# In[10]:


# 关系矩阵图
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(dataset.corr(),vmin=-1,vmax=1,interpolation='none')
fig.colorbar(cax)
plt.show()


# In[11]:


# 分离评估数据集
array=dataset.values
X=array[:,0:60].astype(float)
Y=array[:,60]
validation_size=0.2
seed=7
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)

# 评估算法的基准
num_folds=10
seed=7
scording='accuracy'

# 评估算法-原始数据
models={}
models['LR']=LogisticRegression()
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['NB']=GaussianNB()
models['SVM']=SVC()


# In[12]:


# random.shuffle(True)

# # 默认参数比较算法
# results=[]
# for key in models:
#     kfold=KFold(n_splits=num_folds,random_state=seed)
#     cv_results=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scording)
#     results.append(cv_results)
#     print('%s:%f(%f)'%(key,cv_results.mean(),cv_results.std()))

# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(models[key],X_train,Y_train,cv = kfold,scoring=scording)
    results.append(cv_result)
    print('%s: %f(%f)'%(key,cv_result.mean(),cv_result.std()))


# In[13]:


# 箱线图显示数据的分布状况
fig=plt.figure()
fig.suptitle('Algorithm Comparision')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# In[14]:


# 评估算法-正态化数据
pipelines={}
pipelines['ScalerLR']=Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression())])
pipelines['ScalerLDA']=Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])
pipelines['ScalerKNN']=Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])
pipelines['ScalerCART']=Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier())])
pipelines['ScalerNB']=Pipeline([('Scaler',StandardScaler()),('NB',GradientBoostingClassifier())])
pipelines['ScalerSVM']=Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])
results=[]
for key in pipelines:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_results=cross_val_score(pipelines[key],X_train,Y_train,cv=kfold,scoring=scording)
    results.append(cv_results)
    print('%s:%f(%f)'%(key,cv_results.mean(),cv_results.std()))


# In[15]:


# 箱线图显示数据的分布状况
fig=plt.figure()
fig.suptitle('Scaled Algorithm Comparision')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# In[16]:


# 调参改进算法-KNN
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model=KNeighborsClassifier()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scording,cv=kfold)
grid_result=grid.fit(X=rescaledX,y=Y_train)

print('最优:%s使用%s'%(grid_result.best_score_,grid_result.best_params_))
cv_results=zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
for mean,std,param in cv_results:
    print('%f(%f) with %r'%(mean,std,param))


# In[17]:


# 调参改进算法
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train).astype(float)
param_grid={}
param_grid['C']=[0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0]
param_grid['kernel']=['linear','poly','rbf','sigmoid']
model=SVC()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scording,cv=kfold)
grid_result=grid.fit(X=rescaledX,y=Y_train)

print('最优:%s使用%s'%(grid_result.best_score_,grid_result.best_params_))
cv_results=zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
for mean,std,param in cv_results:
    print('%f(%f) with %r'%(mean,std,param))


# In[18]:


# 集成算法
ensembles={}
ensembles['ScaledAB']=Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostClassifier())])
ensembles['ScaledGBM']=Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingClassifier())])
ensembles['ScaledRF']=Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestClassifier())])
ensembles['ScaledET']=Pipeline([('Scaler',StandardScaler()),('ETR',ExtraTreesClassifier())])

results=[]
for key in ensembles:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_results=cross_val_score(ensembles[key],X_train,Y_train,cv=kfold,scoring=scording)
    results.append(cv_results)
    print('%s:%f(%f)'%(key,cv_results.mean(),cv_results.std()))


# In[19]:


# 集成算法-箱线图
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()


# In[20]:


# 集成算法GBM-调参
# random.shuffle()=True
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
param_grid={'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
model=GradientBoostingClassifier()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scording,cv=kfold)
grid_result=grid.fit(X=rescaledX,y=Y_train)
print('最优:%s使用%s'%(grid_result.best_score_,grid_result.best_params_))


# In[21]:


# 模型最终化
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
model=SVC(C=1.5,kernel='rbf')
model.fit(X=rescaledX,y=Y_train)

# 评估模型
rescaled_validationX=scaler.transform(X_validation)
predictions=model.predict(rescaled_validationX)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

