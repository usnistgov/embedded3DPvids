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


class meshPlot(metricPlot):
    '''contour plot with no interpolation
        xvar is the x variable name, yvar is the y variable name. 
        zvar is the variable to color by. 
        logx, logy to plot on a log scale
        dx>0 to group points by x and take error. otherwise, plot everything. dx=1 to average all points together
        cmapname is the name of the colormap in matplotlib
        '''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, plotReg:bool=False, grid:bool=True
                 , lines:bool=False, dx:float=0.1, dy:float=0.1, legend:bool=True, legendloc:str='right', **kwargs):
        self.plotReg = plotReg
        self.grid = grid
        self.lines = lines
        self.legend = legend
        self.legendloc = legendloc
        super().__init__(ms, ss, dx=dx, dy=dy, **kwargs)
        self.plot()
        
    def roundToOrder(self, val:float) -> float:
        '''round the value to 2 sig figs'''
        if abs(val)<10**-14:
            return 0
        else:
            return round(val, -int(np.log10(abs(val)))+1)
        
    def getMeshTicks(self, x:bool, log:bool) -> Tuple[list,list]:
        '''get clean list of ticks and tick positions from pivot table'''
        if x:
            ylist = self.piv.columns
            n = self.piv.shape[1]
        else:
            ylist = self.piv.index
            n = self.piv.shape[0]
        if len(ylist)==1:
            dy = 1
        else:
            dys = ylist[1:]-ylist[:-1]
            if not log and max(dys)-min(dys)>10^-3:
                ticksf = [0.5+i for i in range(n)]
                labelsf = [round(y,10) for y in ylist]
                return ticksf, labelsf
            dy = ylist[1]-ylist[0]
        y0 = ylist[0] # original y0, where ticks = m(x-0.5)+y0
        m = self.roundToOrder(dy) # clean m
        y0f = round(y0, int(y0/m))
        if log:
            labelsfi = [str(self.roundToOrder(y0f+i*m)) for i in range(n)]
            labelsf = ["$10^{{{}}}$".format(i) for i in labelsfi]
            posf = [y0f+i*m for i in range(n)]
            ticksf = [(y-y0)/m+0.5 for y in posf]
        else:
            
            labelsf = [self.roundToOrder(y0f+i*m) for i in range(n)]
            ticksf = [(y-y0)/m+0.5 for y in labelsf]
        return ticksf, labelsf
    
    def scaleZ(self, df2:pd.DataFrame) -> pd.DataFrame:
        '''scale z to a reasonable order so we can see labels'''
        self.zlabel = self.ms.varSymbol(self.zvar)
        maxval = max(abs(df2.c))
        if maxval>10**-1 and maxval<10**2:
            return df2
        order = int(np.floor(np.log10(maxval)))
        df2.c = df2.c/10**order
        self.zlabel = self.zlabel+'$*10^{'+f'{order}'+'}$'
        return df2
        
        
    def plot(self):
        '''make the plot'''
        self.dropNA()
        self.setLinear()
        df2 = self.toGrid(self.ss, rigid=(self.dx>0 or self.dy>0))
        if self.logx:
            df2['x'] = [np.log10(x) for x in df2['x']]
        if self.logy:
            df2['y'] = [np.log10(y) for y in df2['y']]
        df2 = self.scaleZ(df2)
        self.piv = pd.pivot_table(df2, index='y', columns='x', values='c')
        if 'vmin' in self.kwargs0 and 'vmax' in self.kwargs0:
            self.sc = self.ax.pcolormesh(self.piv, cmap=self.cmapname
                                         , vmin=self.kwargs0['vmin'], vmax=self.kwargs0['vmax'])
        else:
            self.sc = self.ax.pcolormesh(self.piv, cmap=self.cmapname)
        if self.dx>0:
            xpos, xticks = self.getMeshTicks(True, self.logx)
        else:
            xpos = [0.5+i for i in range(len(self.piv.columns))]
            xticks = self.piv.columns
        
        self.ax.set_xticks(xpos, minor=False)
        self.ax.set_xticklabels(xticks, minor=False) 
        if self.dy>0:
            ypos, yticks = self.getMeshTicks(False, self.logy)
        else:
            ypos = [0.5+i for i in range(len(self.piv.index))]
            yticks = self.piv.index
        
        self.ax.set_yticks(ypos, minor=False)
        self.ax.set_yticklabels(yticks, minor=False)
        for i in range(self.piv.shape[0]):
            for j in range(self.piv.shape[1]):
                c = self.piv.iloc[i].iloc[j]
                if not pd.isna(c):
                    self.ax.text(j+0.5,i+0.5,'{:0.2f}'.format(c), horizontalalignment='center',verticalalignment='center')
        if self.legend: 
            cbar = plt.colorbar(self.sc, label=self.zlabel, location = self.legendloc, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=self.fs)
        self.setSquare()