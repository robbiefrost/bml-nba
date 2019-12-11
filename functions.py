import math
import numpy as np
import pandas as pd

def remove_stats(df,stat_list):
    cols = df.columns
    index_list = []
    for c in range(len(cols)):
        for stat in stat_list:
            if stat in cols[c].split('_'):
                index_list.append(c)
    df = df.iloc[:, [j for j, c in enumerate(df.columns) if j not in index_list]]
    return df

def restrict_stats(df,stat_list):
    cols = df.columns
    index_list = []
    for c in range(len(cols)):
        for stat in stat_list:
            if stat in cols[c].split('_'):
                index_list.append(c)
    df = df.iloc[:, [j for j, c in enumerate(df.columns) if j in index_list]]
    return df

def drop_nan(df,col):
    index_list = []
    for i in range(df.shape[0]):
        if math.isnan(df.at[i,col]):
            index_list.append(i)

    df = df.drop(df.index[index_list])
    return df

def get_quarter_features(df,q):

    if q == 0:
        l = [0,1,2,3]
    elif q == 5:
        l = [2,3]
    elif q == 6:
        l = [0,1]
    else:
        l = list(range(4))
        del l[q-1]
    fq = remove_stats(df,[str(k+1) for k in l])
    return fq
