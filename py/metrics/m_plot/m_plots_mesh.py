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
        if log:
            dy = np.log10(ylist[1])-np.log10(ylist[0])
            y0 = np.log10(ylist[0])
        else:
            if len(ylist)==1:
                dy = 1
            else:
                dy = ylist[1]-ylist[0]
            y0 = ylist[0] # original y0, where ticks = m(x-0.5)+y0
        m = self.roundToOrder(dy) # clean m
        if log:
            logy0f = self.roundToOrder(round(np.log10(ylist[0])/m)*m) # clean y0
            labelsfi = [str(self.roundToOrder(logy0f+i*m)) for i in range(n)]
            labelsf = ["$10^{{{}}}$".format(i) for i in labelsfi]
            posf = [logy0f+i*m for i in range(n)]
            ticksf = [(y-y0)/m+0.5 for y in posf]
        else:
            y0f = round(ylist[0], int(ylist[0]/m))
            labelsf = [self.roundToOrder(y0f+i*m) for i in range(n)]
            ticksf = [(y-y0)/m+0.5 for y in labelsf]
        return ticksf, labelsf
        
    def plot(self):
        '''make the plot'''
        self.dropNA()
        df2 = self.toGrid(self.ss, rigid=(self.dx>0 or self.dy>0))
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
        vs = self.ms.varSymbol(self.zvar)
        if self.legend: 
            cbar = plt.colorbar(self.sc, label=vs, location = self.legendloc, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=self.fs)
        self.setSquare()
        # self.ax.set_title(vs, fontsize=self.fs)