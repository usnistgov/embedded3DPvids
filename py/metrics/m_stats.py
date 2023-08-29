#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import numpy as np

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
import tools.regression as rg

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)

#----------------------------------------------
    
def sem(l:list) -> float:
    '''standard error'''
    l = np.array(l)
    l = l[~np.isnan(l)]
    if len(l)==0:
        return np.nan
    return np.std(l)/np.sqrt(len(l))

def msen(l:list) -> Tuple[float, float, int]:
    '''get mean and standard error'''
    l = np.array(l)
    l = l[~np.isnan(l)]
    if len(l)==0:
        return np.nan, np.nan, 0
    m = np.mean(l)
    se = np.std(l)/np.sqrt(len(l))
    return m,se, len(l)

def pooledSESingle(df:pd.DataFrame, var:str) ->  Tuple[float, float]:
    '''calculate the pooled standard error for a group of values, each with their own standard error'''
    if 'xs' in var:
        n = 4
    elif 'vert' in var:
        n = 4
    elif 'horiz' in var:
        n = 3
    mean = df[var].mean()
    sevar = f'{var}_SE'
    if len(df)>1:
        if sevar in df:
            a = np.sum([n*(np.sqrt(n)*row[sevar])**2 for i,row in df.iterrows()])/(n*len(df))
            b = np.sum([n**2*(df.iloc[i][var]-df.iloc[i+1][var])**2 for i in range(len(df)-1)])/(n**2*len(df))
            poolsd = np.sqrt(a+b)
            se = poolsd/np.sqrt(len(df))
        else:
            se = df[var].sem()
    else:
        se = 0
    return mean, se


def pooledSE(vals:list, ses:list, ns:list) -> Tuple[float, float, int]:
    '''given list of means, standard errors, and sizes, calculate the pooled mean, standard error, and sample size for a group of values, each with their own standard error. https://en.wikipedia.org/wiki/Pooled_variance'''
    N = len(vals)   # number of means
    n = np.sum(ns)  # number of samples
    mean = np.sum([(ns[i]*vals[i]) for i in range(N)])/n # weighted mean
    
    if not len(vals)==len(ses) or not len(ses)==len(ns):
        raise ValueError(f'Mismatched array lengths in pooledSE: vals {len(vals)}, SE {len(ses)}, N {len(ns)}')
       
    
    if len(ses)>1:
        if n==N:
            se = np.std(vals, ddof=1) / np.sqrt(np.size(vals))
        else:
            ss = [ses[i]*np.sqrt(ns[i]) for i in range(N)]  # standard deviations
            a = np.sum([(ns[i]-1)*ss[i]**2 + (ns[i]*vals[i]**2) for i in range(N)])
            b = n*mean**2
            poolsd = np.sqrt((a-b)/(n-N))
            se = poolsd/np.sqrt(n)
    elif len(ses)==1:
        se = ses[0]
    else:
        se = vals.sem()
    return mean, se, n

def pooledSEDF(df:pd.DataFrame, var:str) ->  Tuple[float, float]:
    '''given a dataframe and a variable name, calculate the pooled mean, standard error, and sample size for a group of values, each with their own standard error'''
    sevar = f'{var}_SE'
    nvar = f'{var}_N'
    if not nvar in df:
        if 'xs' in var or 'vert' in var:
            df.loc[:,nvar] = 4
        elif 'horiz' in var:
            df.loc[:,nvar] = 3
        else:
            if not sevar in df:
                vals = df[var].dropna()
                return vals.mean(), vals.sem(), len(vals)
            else:
                raise ValueError(f'Cannot determine N for {var}')
    df2 = df[[var, sevar, nvar]].dropna()
    vals = list(df2[var])
    ses = list(df2[sevar])
    ns = list(df2[nvar])
    return pooledSE(vals, ses, ns)

def tossBigSE(df:pd.DataFrame, column:str, quantile:float=0.9):
    '''toss rows with big standard errors from the list'''
    if not column[-3:]=='_SE':
        column = f'{column}_SE'
    return df[df[column]<df[column].quantile(quantile)]

