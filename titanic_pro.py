# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:52:07 2018

@author: yanchao
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
import os
import warnings
from sklearn.preprocessing import LabelEncoder
#----------------------------------------函数部分-------------------------------------
# one-hot encoding
def my_one_hot(var_name,raw_data,new_data):
    import numpy as np
    import pandas as pd
    data0=pd.get_dummies(raw_data[var_name])
    n=np.shape(data0)[1]
    columns=list()
    for i in range(n):
        columns.append(var_name+"_"+str(i+1))
    new_data[columns]=data0
    if var_name in new_data.columns.tolist():
        del new_data[var_name]
    return 0

def missing_values_table(df):
    # Total missing values
    import pandas as pd
    mis_val = df.isnull().sum()
        
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
    "There are " + str(mis_val_table_ren_columns.shape[0]) +
    " columns that have missing values.")
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product
 
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
 
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
 
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None
 
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
 
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
 
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()
 
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
 
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
 
        return nf_train, nf_test, prior, col_avg_y
 
    def fit_transform(self, X, y):
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)
 
        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
 
    def transform(self, X):
        X_new = X.copy()
 
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
 
        return X_new

warnings.filterwarnings('ignore')
os.chdir("D:\\kaggle\\titanic")

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
total=pd.concat([train,test],axis=0)

##-------------------------------------缺失值处理---------------------------------
missing_values_table(total) # 从结果可以看出，除了Survived以外，缺失的变量为Cabin,Age,Embarked,Fare
## Embarked缺失部分分别只有2个，用众数填补
total["Embarked"].loc[total["Embarked"].isna()]=total["Embarked"].mode().iloc[0]
total["Ticket"].drop_duplicates() # 从这里可以看出，Fare对应的是一个或多个ticket
fare_dict_size=dict(total["Fare"].groupby(total["Ticket"]).count())
total["size"]=total["Ticket"].apply(lambda x: fare_dict_size[x])
total["Fare_adj"]=total["Fare"]/total["size"]

# 注意到Fare_adj和Fare有一个数据缺失，现对其利用embarked和Pclass平均值进行补全
total["Fare_adj"].loc[total["Fare_adj"].isna()]=total["Fare_adj"].loc[(total["Embarked"]=="S")&(total["Pclass"]==1)].mean()
total["Fare"].loc[total["Fare"].isna()]=total["Fare_adj"].loc[total["Fare"].isna()]*total["Fare_adj"].loc[total["Fare"].isna()]

# Age数据缺失部分用其他的变量进行补充，首先看一下Age与其他变量之间的关系
corr_data=total.corr()
# 从结果可以看出，Age和P_class,Fare_adj,Sibsp，size以及Parch相关系数很高，size近似于SibSp和Parch之和，因此我们利用Pclass,Fare_adj,Sibsp和Parch

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(total[["Pclass","Fare_adj","SibSp","Parch"]].loc[~total["Age"].isna()],total["Age"].loc[~total["Age"].isna()])
total["Age"].loc[total["Age"].isna()]=lm.predict(total[["Pclass","Fare_adj","SibSp","Parch"]].loc[total["Age"].isna()])
# 将回归得到的Age小于0的部分转化为1
total["Age"].loc[total["Age"]<0]=1

# Cabin: 按照开头字母分类
total["First_alp"]="XXX"
total["First_alp"].loc[~total["Cabin"].isna()]=total["Cabin"].loc[~total["Cabin"].isna()].apply(lambda x:x[0])
del total["Cabin"]

# 到此,缺失值全部处理完毕，接下来开始特征工程。

#--------------------------------------特征工程-----------------------------------------
## Name 处理
# 首先提取title
total["title"]=total["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
# 然后提取姓
total["first_name"]=total["Name"].apply(lambda x: x.split(",")[1].split(".")[1])
# 考察姓氏不同的个数:len(total["first_name"].drop_duplicates()),结果为1126
total["first_name_code"]=LabelEncoder().fit(total["first_name"]).transform(total["first_name"])
total["title_code"]=LabelEncoder().fit(total["title"]).transform(total["title"])
# 删除原始数据
del total["Name"],total["first_name"],total["title"]

## Ticket 处理
# 观察一下total["Ticket"]的各种类型
# 按照Ticket的重复情况直接做Ticket_code
total["Ticket_code"]=LabelEncoder().fit(total["Ticket"]).transform(total["Ticket"])
del total["Ticket"]

# 对Embarked，Sex，First_alp，title_code直接做one_hot编码
for i in ["Embarked","Sex","First_alp","title_code","Pclass","first_name_code","Ticket_code"]:
    my_one_hot(i,total,total)

# 分开train和test数据
train=total[:891]
test=total[891:]
y=train.Survived
train=train.drop(["Survived","PassengerId"],axis=1)
test=test.drop(["Survived","PassengerId"],axis=1)

#------------------------------Random Forest------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(train,y)

y_pred=rf_model.predict(test)
submission=pd.read_csv("gender_submission.csv")
submission.Survived=y_pred
submission.to_csv("rf_pred.csv",header=True,index=False)
# 最终结果为0.79904, 提交时排名为1588/10011












































































































































