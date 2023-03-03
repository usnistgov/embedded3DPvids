#!/usr/bin/env python
'''tools for fitting regressions'''

# external packages
import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#----------------------------------------------


def polyfit(x:List[float], y:List[float], degree:int) -> dict:
    '''fit polynomial'''
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r2'] = ssreg / sstot
    results['coeffs'] = list(coeffs)

    return results

def quadReg(x:list, y:list) -> dict:
    '''quadratic regression, where x and y are lists'''
    res = polyfit(x,y,2)
    return {'a':res['coeffs'][0], 'b':res['coeffs'][1], 'c':res['coeffs'][2], 'r2':res['r2']}

def polyMultiFit(X:np.array, y:np.array, order, intercept:Union[float,str]) -> dict:
    '''polynomial fit for multiple regression'''
    poly = PolynomialFeatures(degree = order)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    return lr(X_poly, y, intercept)

def lr(X:np.array, y:np.array, intercept:Union[float,str]) -> dict:
    '''linear regression from numpy arrays'''
    if type(intercept) is str:
        regr = LinearRegression().fit(X,y)
    else:
        y = y-intercept
        regr = LinearRegression(fit_intercept=False).fit(X,y)
    rsq = regr.score(X,y)
    c = regr.intercept_
    if not type(intercept) is str:
        c = c+intercept
    b = regr.coef_
    out = {'c':c, 'r2':rsq}
    if X.shape[1]==1:
        out['b'] = b[0]
    else:
        for i,bi in enumerate(b):
            out[f'b{i}'] = bi
    return out

def linearReg(x:list, y:list, intercept:Union[float, str]='') -> dict:
    '''Get a linear regression from lists. y=bx+c'''
    if len(y)<5:
        return {}
    y = np.array(y)
    X = np.array(x).reshape((-1,1))
    return lr(X,y,intercept)
    

def regPD(df:pd.DataFrame, xcols:List[str], ycol:str, order:int=1, intercept:Union[float,str]='') -> dict:
    '''linear regression from pandas dataset'''
    df2 = df.dropna(subset = xcols+[ycol]) # remove any NaNs from the columns of interest
    y = df2[ycol]
    X = df2[xcols]
    if order==1:
        return lr(X,y,intercept)
    else:
        return polyMultiFit(X,y,order, intercept)
    
    
def spearman(df:pd.DataFrame, xcol:str, ycol:str) -> dict:
    '''get spearman rank correlation'''
    ssi = df.dropna(subset=[xcol, ycol]) # drop na in colums
    corr, p = stats.spearmanr(ssi[xcol], ssi[ycol])
    return {'spearman_corr':corr, 'spearman_p':p}
    
def removeOutliers(df:pd.DataFrame, col:str, sigma:float=3) -> pd.DataFrame:
    '''remove outliers in column by # of standard deviation sigma'''
    return df[np.abs(df[col]-df[col].mean()) <= (sigma*df[col].std())]
    
