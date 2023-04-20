#!/usr/bin/env python
'''Functions for plotting still and video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
import matplotlib.colors as mc
import colorsys
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import seaborn as sns
import string
from scipy import stats
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from m_plots import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')



#-------------------------------------------------------------

def simplifyType(s:Union[str, pd.DataFrame]):
    '''if given a dataframe, simplify the sweepType. otherwise, simplify the individual string'''
    if type(s) is str:
        # just take first 2 elements
        spl = re.split('_', s)
        return f'{spl[0]}_{spl[1]}'
    else:
        # convert all strings in sweepType column
        s.loc[:,'sweepType'] = [simplifyType(si) for si in s['sweepType']]
        
def speedSweeps(ss:pd.DataFrame) -> pd.DataFrame:
    '''only get the speedsweeps'''
    return ss[(ss.sweepType.str.startswith('speed'))|(ss.sweepType.str.startswith('$v$ sweep'))|(ss.sweepType.str.startswith('$v$, '))]

def viscSweeps(ss:pd.DataFrame) -> pd.DataFrame:
    return ss[ss.sweepType.str.startswith('visc')|(ss.sweepType.str.startswith('$\\eta$ sweep'))|(ss.sweepType.str.startswith('$\\eta$, '))]
    

def sweepTypeSS(ms:summaryMetric, xvar:str, yvar:str, **kwargs):
    '''plot values based on sweep type'''
    sp0 = scatterPlot(ms, xvar=xvar, yvar=yvar, **kwargs)
    ss = ms.ss
    ss.sort_values(by='sigma')
    kwargs0 = **kwargs
    kwargs0['ax'] = sp0.ax
    kwargs0['fig'] = sp0.fig
    if len(ss.sigma.unique())==1:
        # all the same surface tension: make visc blue and speed red
        for i,ss0 in enumerate([speedSweeps(ss), viscSweeps(ss)]):
            # iterate through viscosity and speed sweeps
            color0 = sp0.cmap(0.99*i)
            u = ss0.sweepType.unique()
            for j,st in enumerate(u):
                # iterate through unique sweep types
                color = sp0.cmap(1*i + (0.4-i)*j/len(u))
                ma = sp0.getMarker(j, u, color)
                kwargs1 = {**kwargs0, **ma}
                scatterPlot(ss0[ss0.sweepType==st], xvar=xvar, yvar=yvar, zvar='sweepType', **kwargs1)
                if 'yideal' in kwargs0:
                    kwargs0.pop('yideal')
                if 'xideal' in kwargs0:
                    kwargs0.pop('xideal')
    else:
        scatterPlot(speedSweeps(ss), xvar=xvar, yvar=yvar, zvar='sweepType', color='#555555', **kwargs0)
        if 'yideal' in kwargs0:
            kwargs0.pop('yideal')
        if 'xideal' in kwargs0:
            kwargs0.pop('xideal')
        scatterPlot(viscSweeps(ss), xvar=xvar, yvar=yvar, zvar='sweepType',**kwargs0)
    return sp0.fig, sp0.ax



