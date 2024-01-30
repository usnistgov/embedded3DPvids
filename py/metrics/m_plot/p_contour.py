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
from p_metric import *

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


class contourPlot(metricPlot):
    '''contour plot with interpolation
        xvar is the x variable name, yvar is the y variable name. 
        zvar is the variable to color by. 
        logx, logy to plot on a log scale
        dx>0 to group points by x and take error. otherwise, plot everything. dx=1 to average all points together
        cmapname is the name of the colormap in matplotlib
        '''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, plotReg:bool=False, grid:bool=True, lines:bool=False, dx:float=0.1, dy:float=0.1, **kwargs):
        self.plotReg = plotReg
        self.grid = grid
        self.lines = lines
        super().__init__(ms, ss, dx=dx, dy=dy, **kwargs)
        
        self.plot()
        
    def plot(self):
        '''contour plot with interpolation. zvar is color variable'''
        X_unique = np.sort(self.ss[self.xvar].unique())
        Y_unique = np.sort(self.ss[self.yvar].unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = self.ss.pivot_table(index=self.xvar, columns=self.yvar, values=self.cvar).T.values
        zmin = self.ss[self.cvar].min()
        zmax = self.ss[self.cvar].max()
        zmin = round(zmin, -int(np.log10(abs(zmin)))+1)
        zmax = round(zmax, -int(np.log10(abs(zmax)))+1)
        dz = (zmax-zmin)/10
        dz = round(dz, -int(np.log10(abs(dz)))+1)
        levels = np.array(np.arange(zmin, zmax, dz))
        self.sc = self.ax.contourf(X,Y,Z, len(levels), cmap=self.colors.cname)
        line_colors = ['black' for l in self.sc.levels]
        cp = self.ax.contour(X, Y, Z, levels=levels, colors=line_colors)
        self.ax.clabel(cp, fontsize=self.fs, colors=line_colors)
        self.setSquare()
        vs = self.ms.varSymbol(self.cvar)
        if not ('legend' in self.kwargs0 and not self.kwargs0['legend']): 
            cbar = plt.colorbar(self.sc, label=vs, ax=self.ax)
            cbar.ax.tick_params(labelsize=self.fs)
        self.setSquare()
        # self.ax.set_title(vs, fontsize=self.fs)