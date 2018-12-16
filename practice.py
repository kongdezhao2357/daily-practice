#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

x,y=make_hastie_10_2(random_state=1)
data=np.hstack((x,y.reshape((len(y),1))))
# np.savetxt('data.csv',data,delimiter=',')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1)
def default():
    auc_score=[]
    accuracy=[]
    clf=XGBClassifier()
    clf.fit(x_train,y_train)
    y_pre=clf.predict(x_test)#预测最终结果
    y_pro_1=clf.predict_proba(x_test)[:,-1]#预测是1的可能性，浮点
    print('AUC score:%f'%metrics.roc_auc_score(y_test,y_pro_1))#x轴：真正例率，y：假正例率
    print('accuracy:%f'%metrics.accuracy_score(y_test,y_pre))
    auc_score.append(metrics.roc_auc_score(y_test,y_pro_1))
    accuracy.append(metrics.roc_auc_score(y_test,y_pre))

def adjust():
    clf=XGBClassifier(
        learning_rate=.2,
        n_estimators=800,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1
    )
    clf.fit(x_train,y_train)
    y_pre=clf.predict(x_test)
    y_pro=clf.predict_proba(x_test)[:,-1]
    print('auc_score:%f'%metrics.roc_auc_score(y_test,y_pro))
    print('accuracy:%f'%metrics.accuracy_score(y_test,y_pre))

def adjust1():
    clf = XGBClassifier(
        learning_rate=.2,
        n_estimators=800,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1
    )
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    y_pro = clf.predict_proba(x_test)[:, -1]
    print('auc_score:%f' % metrics.roc_auc_score(y_test, y_pro))
    print('accuracy:%f' % metrics.accuracy_score(y_test, y_pre))


if __name__=='__main__':
    # default()
    # AUCscore: 0.969797
    # accuracy: 0.896667
    # adjust()
    # auc_score: 0.989008
    # accuracy: 0.940667
    # adjust1()
    # auc_score: 0.986432
    # accuracy: 0.935667


