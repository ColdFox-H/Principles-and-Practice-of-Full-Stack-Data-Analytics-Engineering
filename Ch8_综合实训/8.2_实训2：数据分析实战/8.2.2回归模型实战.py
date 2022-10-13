#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入类库
import pandas as pd
import numpy as np
import matplotlib as mpl
from numpy import arange
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# 导入数据
filename = 'housing.csv'
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
dataset = pd.read_csv(filename,names= names, delim_whitespace=True)


# In[3]:


# 数据维度
print(dataset.shape)


# In[4]:


# 特征属性的字段类型
print(dataset.dtypes)


# In[5]:


# 描述统计信息
pd.set_option('precision',1)
print(dataset.describe())


# In[6]:


# 关联关系
pd.set_option('precision',2)
print(dataset.corr(method='pearson'))


# In[7]:


# 直方图
dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
plt.show()


# In[8]:


# 密度图
dataset.plot(kind = 'density', subplots = True, layout = (4,4),sharex = False, fontsize =1)
plt.show()


# In[9]:


# 箱线图
dataset.plot(kind ='box', subplots = True, layout = (4,4),sharex = False, sharey = False,fontsize = 8)
plt.show()


# In[10]:


# 散点矩阵图
scatter_matrix(dataset)
plt.show()


# In[11]:


# 相关矩阵图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(),vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[12]:


# 分离数据集
array = dataset.values
X = array[:, 0:13]
Y = array[:, 13]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state= seed)


# In[13]:


# 评估算法 —— 评估标准
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


# In[14]:


# 评估算法 —— baseline
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] =DecisionTreeRegressor()
models['SVM'] = SVR()


# In[15]:


# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(models[key],X_train,Y_train,cv = kfold,scoring=scoring)
    results.append(cv_result)
    print('%s: %f(%f)'%(key,cv_result.mean(),cv_result.std()))


# In[16]:


# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# In[17]:


# 评估算法 —— 正太化数据
pipelines = {}
pipelines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
pipelines['ScalerLASSO'] = Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])
pipelines['SclaerEn'] = Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])
pipelines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
pipelines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])
pipelines['ScalerSVM'] = Pipeline([('Scaler',StandardScaler()),('SVM',SVR())])
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(models[key],X_train,Y_train,cv = kfold,scoring=scoring)
    results.append(cv_result)
    print('%s: %f(%f)'%(key,cv_result.mean(),cv_result.std()))


# In[18]:


# 评估算法 —— 箱线图
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# In[19]:


# 调参改善算法 —— KNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model, param_grid = param_grid, scoring=scoring,cv=kfold)
grid_result = grid.fit(X = rescaledX, y=Y_train)
print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                grid_result.cv_results_['std_test_score'],
                grid_result.cv_results_['params'])
for mean, std, param in cv_results :
    print('%f(%f) with %r'%(mean,std,param))


# In[20]:


# 集成算法
ensembles = {}
ensembles['ScaledAB'] = Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
ensembles['ScaledAB-KNN'] = Pipeline([('Scaler',StandardScaler()),('ABKNN',AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScaledAB-LR'] = Pipeline([('Scaler',StandardScaler()),('ABLR',AdaBoostRegressor(LinearRegression()))])
ensembles['ScaledRFR'] = Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestRegressor())])
ensembles['ScaledETR'] = Pipeline([('Scaler',StandardScaler()),('ETR',ExtraTreesRegressor())])
ensembles['ScaledGBR'] = Pipeline([('Scaler',StandardScaler()),('GBR',GradientBoostingRegressor())])

results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(ensembles[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s: %f(%f)'%(key,cv_result.mean(),cv_result.std()))


# In[21]:


# 集成算法 —— 箱线图
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()


# In[22]:


# 集成算法GBM —— 调参
caler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
model = GradientBoostingRegressor()
kfold = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv= kfold)
grid_result = grid.fit(X = rescaledX, y = Y_train)
print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))

# 集成算法ET —— 调参
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators':[5,10,20,30,40,50,60,70,80,90,100]}
model = ExtraTreesRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid = param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X = rescaledX,y =Y_train)
print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))


# In[23]:


# 训练模型
caler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
gbr = ExtraTreesRegressor(n_estimators=80)
gbr.fit(X = rescaledX,y = Y_train)


# In[24]:


# 评估算法模型
rescaledX_validation = scaler.transform(X_validation)
predictions = gbr.predict(rescaledX_validation)
print(mean_squared_error( Y_validation,predictions))


# In[ ]:




