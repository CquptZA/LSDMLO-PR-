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
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelO
from skmultilearn.adapt import MLTSVM
from skmultilearn.adapt import MLARAM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from enum import Enum
import sklearn.metrics as metrics
from scipy import sparse
from skmultilearn.model_selection import IterativeStratification
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
warnings.filterwarnings("ignore")
# Imbalance and need function

def CardAndDens(X,y):
    cardmatrix=[]
    for i in range(X.shape[0]):
        count=0
        for j in range(y.shape[1]):
            if y[i,j]==1:
                count+=1
        cardmatrix.append(count)
    Card=sum(cardmatrix)/len(cardmatrix)
    Dens=Card/y.shape[1]
    return Card,Dens
def FeatureSelect(p):
    if p==1:
        return X.toarray(),feature_names
    else:
        featurecount=int(X.shape[1]*p)
        Selectfeatureindex=[x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]),key=lambda x: x[1],reverse=True))][:featurecount]
        Allfeatureindex=[i for i in range(X.shape[1])]
        featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
        new_x=np.delete(X.toarray(),featureindex,axis=1)
        new_featurename=[feature_names[i] for i in Selectfeatureindex] 
        return new_x,new_featurename
def ImR(X,y):
    Imr=[]
    for i in range(y.shape[1]):
        count0=0
        count1=0
        for j in range(y.shape[0]):
            if y[j,i]==1:
                count1+=1
            else:
                count0+=1
        if count1<=count0:
            Imr.append(count0/count1)
        else:
            Imr.append(count1/count0)
    return Imr
def Imbalance(X,y):
    countmatrix=[]
    for i in range(y.shape[1]):
        count0=0
        count1=0
        for j in range(y.shape[0]):
            if y[j,i]==1:
                count1+=1
            else:
                count0+=1
        countmatrix.append(count1)
    maxcount=max(countmatrix)
    ImbalanceRatioMatrix=[maxcount/i for i in countmatrix]
    MaxIR=max(ImbalanceRatioMatrix)
    MeanIR=sum(ImbalanceRatioMatrix)/len(ImbalanceRatioMatrix)
    return ImbalanceRatioMatrix,MeanIR,countmatrix
def Scumble(X,y):
    ImbalanceRatioMatrix,MeanIR,_=Imbalance(X,y)
    DifferenceImbalanceRatioMatrix=[i-MeanIR for i in ImbalanceRatioMatrix]
    count=0
    for i in range(y.shape[1]):
        count+=math.pow(DifferenceImbalanceRatioMatrix[i],2)/(y.shape[1]-1)
    ImbalanceRatioSigma=math.sqrt(count)
    CVIR=ImbalanceRatioSigma/MeanIR
    SumScumble=0
    Scumble_i=[]
    for i in range(y.shape[0]):
        count=0
        prod=1
        SumIRLbl=0
        for j in range(y.shape[1]):
            IRLbl=1
            if y[i,j]==1:
                IRLbl=ImbalanceRatioMatrix[j]
                SumIRLbl+=IRLbl
                prod*=IRLbl
                count+=1
        if count==0:
            Scumble_i.append(0)
        else:
            IRLbl_i=SumIRLbl/count
            Scumble_i.append(1.0-((1.0/IRLbl_i) * math.pow(prod, 1.0/count)))
    scumble=sum(Scumble_i)/X.shape[0]
    return Scumble_i,scumble,CVIR
def CalcuNN(df1,n_neighbor):
    nbs=NearestNeighbors(n_neighbors=n_neighbor,metric='euclidean',algorithm='kd_tree').fit(df1)
    euclidean,indices= nbs.kneighbors(df1)
    return euclidean,indices
def FeatureSelect(p,feature_names,X):
    if p==1:
        return X.toarray(),feature_names
    else:
        if feature_names[1][1]=='NUMERIC':
            featurecount=int(X.shape[1]*p)
            column_variances = np.var(X.toarray(), axis=0)
            sorted_indices = column_variances.argsort()[::-1]
            Selectfeatureindex = sorted_indices[:featurecount]
            Allfeatureindex=[i for i in range(X.shape[1])]
            featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
            new_x=np.delete(X.toarray(),featureindex,axis=1)
            new_featurename=[feature_names[i] for i in Selectfeatureindex]          
        else:
            featurecount=int(X.shape[1]*p)
            Selectfeatureindex=[x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]),key=lambda x: x[1],reverse=True))][:featurecount]
            Allfeatureindex=[i for i in range(X.shape[1])]
            featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
            new_x=np.delete(X.toarray(),featureindex,axis=1)
            new_featurename=[feature_names[i] for i in Selectfeatureindex] 
        return new_x,new_featurename
def LabelSelect(label_names,y):
    b=[]
    new_labelname=[i for i in label_names]
    for i in range(y.shape[1]):
        if y[:,i].sum()<=20:
            b.append(i)
            new_labelname.remove(label_names[i])
    new_y=np.delete(y.toarray(),b,axis=1)
    return new_y,new_labelname 
