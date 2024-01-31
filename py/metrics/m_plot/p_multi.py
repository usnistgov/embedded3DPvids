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
import tools.regression as rg
from tools.config import cfg
from tools.figureLabels import subFigureLabels
from sizes import sizes

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

class multiPlot:
    '''for plotting multiple axes in one figure'''
    
    def __init__(self, rows:int=1, cols:int=1, plotType:str='ppt'
                 , tightLayout:bool=True, sharex:bool=False, sharey:bool=False
                 , set_xlabels:bool=True, set_ylabels:bool=False
                 , setSquare:bool=False, legendAbove:bool=False, **kwargs):
        self.rows = rows
        self.cols = cols
        self.plotType = plotType
        self.sharex = sharex
        self.sharey = sharey
        self.set_xlabels = set_xlabels
        self.set_ylabels = set_ylabels
        self.setSquare = setSquare
        self.tightLayout = tightLayout
        self.legendAbove = legendAbove
        self.kwargs0 = kwargs
        self.setUpDims()

        
    def setUpDims(self):
        '''set up the plot'''
        self.fs, self.figsize, self.markersize, self.linewidth = sizes(self.rows, self.cols, self.plotType).values()
        if 'figsize' in self.kwargs0:
            self.figsize = self.kwargs0['figsize']
        plt.rc('font', size=self.fs) 

        if self.legendAbove:
            self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
            widths = [1 for i in range(self.cols)]
            heights = [1]+[10 for i in range(self.rows)]
            gs = self.fig.add_gridspec(ncols=self.cols, nrows=self.rows+1,
                                      height_ratios=heights)
            self.legendAx =  self.fig.add_subplot(gs[0, :])  # put legend on top row
            self.axs = np.array([[self.fig.add_subplot(gs[j+1,i]) for i in range(self.cols)] for j in range(self.rows)])
        else:
            self.fig, self.axs = plt.subplots(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, figsize=self.figsize)
        if self.rows==1 or self.cols==1:
            self.axs = np.reshape(self.axs, (self.rows, self.cols))

        self.axlabels = [(i,j) for i in range(self.rows) for j in range(self.cols)]
        self.objs = np.empty((self.rows, self.cols), dtype=object)
        self.ys = np.array([['' for i in range(self.cols)] for j in range(self.rows)], dtype='object')  # array of y variables
        self.xs = np.array([['' for i in range(self.cols)] for j in range(self.rows)], dtype='object')  # array of x variables

        
    def applyOverAxes(self, func) -> None:
        '''run a function on all axes'''
        for i in range(self.rows):
            for j in range(self.cols):
                ax = self.axs[i,j]
                func(i,j,ax)
                
    def shareAxes(self, i1:int, j1:int, i2:int, j2:int, aname:str):
        '''share the x or y axes between 2 axes at coords i1,j1 and i2,j2'''
        if aname=='x':
            self.axs[i1,j1].sharex(self.axs[i2,j2])
        else:
            self.axs[i1,j1].sharey(self.axs[i2,j2])
                
    def setlim(self, i:int, j:int, ax, lv:str) -> None:
        '''set the limit of the axis. lv is the axis variable name, x or y'''
        xlim = self.kwargs0[f'{lv}lim']
        if type(xlim) is dict:
            lim = xlim[self.xs[i,j]]
        else:
            lim = xlim
        if lv=='x':
            ax.set_xlim(lim)
        else:
            ax.set_ylim(lim)
        
    def setxlim(self, i:int, j:int, ax) -> None:
        self.setlim(i,j,ax,'x')
        
    def setylim(self, i:int, j:int, ax) -> None:
        self.setlim(i,j,ax,'y')
        
    def clean(self):
        '''clean up the plots'''
        # if self.setSquare:
        #     for objrow in self.objs:
        #         for obj in objrow:
        #             obj.setSquare()
        if self.plotType=='paper':
            axl = np.array([self.axs[i] for i in self.axlabels])
            subFigureLabels(axl, horiz=True, inside=False)
        if 'xlim' in self.kwargs0:
            self.applyOverAxes(self.setxlim)
        if 'ylim' in self.kwargs0:
            self.applyOverAxes(self.setylim)
        # if self.set_xlabels:
        #     for ax in self.fig.axes:
        #         ax.xaxis.set_tick_params(labelbottom=True)
        for objrow in self.objs:
            for obj in objrow:
                if not obj is None:
                    obj.fixTicks()
        if self.tightLayout:
            self.fig.tight_layout()
                
    def export(self, fn:str):
        if '.' in fn:
            fn = re.split('.', fn)[0]
        for ext in ['.svg', '.png']:
            self.fig.savefig(f'{fn}{ext}', bbox_inches='tight', dpi=300)
        logging.info(f'Exported {fn}.png and .svg')
        
    def pTitle(self, y:str, killWritten:bool) -> str:
        '''title for a y variable type'''
        yy = self.ms.varSymbol(y)
        if killWritten:
            yy = yy.replace('written ', '')
            yy = yy.replace('disturbed ', '')
            yy = yy.replace('write ', '')
            yy = yy.replace('disturb ', '')
        yy = yy.replace('1st ', '')
        yy = yy.replace('2nd ', '')
        yy = yy.replace('3rd ', '')
        u = self.ms.u[y]
        if len(u)>0:
            yy = f'{yy} ({u})'
        return yy
        