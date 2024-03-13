#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import pdb
import random
import time
import pandas as pd
import pdb
import numpy as np
import itertools
import os
from skmultilearn.model_selection import IterativeStratification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import BRkNNbClassifier
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import load_from_arff
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from enum import Enum
import sklearn.metrics as metrics
from scipy import sparse
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.dataset import load_from_arff
from skmultilearn.ensemble import RakelD
from xgboost import XGBClassifier
from sklearn import model_selection, preprocessing
import scipy
import warnings
from Get_W import *
from Sampling import *
from Metric import *
# traning
# Only the call to MLkNN is provided here. You can add other methods and control them through c_idx
def training(X,y,c_idx,sp,optmParameter,feature_names,label_names):
    Randomlist=[7,20,50,21,72]
    Macro_F=[]
    Micro_F=[]
    Hamming_loss=[]
    Ranking_loss=[]
    One_error=[]
    Macro_AUC=[]
# 2-FOLD
    for i in Randomlist:  
        k_fold = IterativeStratification(n_splits=2,order=1,random_state=42)
        j=0
        for train,test in k_fold.split(X,y):
            classifier =MLkNN(k=10)
            dfx=pd.DataFrame(X[train],columns=[x[0] for x in feature_names])
            dfy=pd.DataFrame(y[train],columns=[x[0] for x in label_names])
            X1, y1 = X[train], y[train]
            W = GetW(X1, y1, optmParameter) 
            new_X, new_y = LSDMLOsampling(dfx, dfy, W, sp,feature_names,label_names)
            X1,y1=np.array(new_X),np.array(new_y)
            classifier.fit(X1,y1)
            X2,y2=np.array(X[test]),np.array(y[test])
            ypred = classifier.predict(X2)
            if scipy.sparse.issparse(ypred):
                ypred = ypred.toarray()
            yprob = classifier.predict_proba(X2)
            if scipy.sparse.issparse(yprob):
                yprob = yprob.toarray()
            Macro_F.append(metrics.f1_score(y2, ypred,average='macro'))
            Micro_F.append(metrics.f1_score(y2, ypred,average='micro'))
            Ranking_loss.append(ranking_loss(y2, ypred, yprob))                     
            Macro_AUC.append(macro_averaging_auc(y2, ypred, yprob))  
            Hamming_loss.append(metrics.hamming_loss(y2, ypred)) 
            One_error.append(one_error(y2, ypred, yprob))

    means = np.array([
    np.mean(Macro_F),
    np.mean(Micro_F),
    np.mean(Macro_AUC),
    np.mean(Ranking_loss),
    np.mean(Hamming_loss),
    np.mean(One_error)
    ])

    stds = np.array([
        np.std(Macro_F),
        np.std(Micro_F),
        np.std(Macro_AUC),
        np.std(Ranking_loss),
        np.std(Hamming_loss),
        np.std(One_error)
    ])
    rounded_means = np.round(means, 4)
    rounded_stds = np.round(stds, 4)
    print(tuple(rounded_means) + tuple(rounded_stds))

