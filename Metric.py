#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np


def macro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(l)
    q = np.sum(Y, 0)

    zero_column_count = np.sum(q == 0)
#     print(f"all zero for label: {zero_column_count}")
    r, c = np.nonzero(Y)
    for i, j in zip(r, c):
        p[j] += np.sum((Y[ : , j] < 0.5) * (O[ : , j] <= O[i, j]))

    i = (q > 0) * (q < n)

    return np.sum(p[i] / (q[i] * (n - q[i]))) / l
def hamming_loss(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = np.sum(Y, 1)
    s2 = np.sum(P, 1)
    ss = np.sum(Y * P, 1)

    return np.sum(s1 + s2 - 2 * ss) / (n * l)
def one_error(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2

    i = np.argmax(O, 1)

    return np.sum(1 - Y[range(n), i]) / n
def ranking_loss(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(n)
    q = np.sum(Y, 1)

    r, c = np.nonzero(Y)
    for i, j in zip(r, c): 
        p[i] += np.sum((Y[i, : ] < 0.5) * (O[i, : ] >= O[i, j]))

    i = (q > 0) * (q < l)

    return np.sum(p[i] / (q[i] * (l - q[i]))) / n

