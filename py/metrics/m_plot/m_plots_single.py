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
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from m_plots_metric import *
from m_plots_scatter import *

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
    
    
class sweepTypeSS(scatterPlot):
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvar:str, yvar:str, **kwargs):
        '''plot values based on sweep type'''
        super().__init__(ms, ss, xvar=xvar, yvar=yvar, zvar='sweepType', gradColor=colorModes.discreteZvar, **kwargs)
        
    def plotSweep(self, i:int, ss0:pd.DataFrame) -> None:
        # iterate through viscosity and speed sweeps
        color0 = self.cmap(0.99*i)
        u = ss0.sweepType.unique()
        for j,st in enumerate(u):
            # iterate through unique sweep types
            color = self.cmap(1*i + (0.4-i)*j/len(u))
            ma = self.getMarker(j, color)
            kwargs1 = {**self.kwargs0, **ma}
            self.plotSeries(ss0[ss0.sweepType==st], kwargs1)
            if 'yideal' in self.kwargs0:
                self.kwargs0.pop('yideal')
            if 'xideal' in self.kwargs0:
                self.kwargs0.pop('xideal')
        
    def plot(self):
        '''make the plot'''
        self.dropNA()   # drop nan values
        self.ss.sort_values(by='sigma')   # sort by sigma
        
        if len(self.ss.sigma.unique())==1:
            # all the same surface tension: make visc blue and speed red
            for i,ss0 in enumerate([speedSweeps(ss), viscSweeps(ss)]):
                self.plotSweep(i,ss0)
        else:
            ma = self.getMarker(0, '#555555')
            kwargs1 = {**self.kwargs0, **ma}
            self.plotSeries(speedSweeps(ss), kwargs1)
            if 'yideal' in self.kwargs0:
                kwargs0.pop('yideal')
            if 'xideal' in kwargs0:
                kwargs0.pop('xideal')
            ma = self.getMarker(1, '#ffffff')
            kwargs1 = {**self.kwargs0, **ma}
            self.plotSeries(viscSweeps(ss), kwargs1)
            
        # add verticals, horizontals
        self.idealLines()

        # add legends
        self.addLegends()      

        # set square
        self.setSquare()
        self.fixTicks()
        if self.plotReg:
            self.regressionSS()
