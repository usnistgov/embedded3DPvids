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
                 , setSquare:bool=False, **kwargs):
        self.rows = rows
        self.cols = cols
        self.plotType = plotType
        self.sharex = sharex
        self.sharey = sharey
        self.set_xlabels = set_xlabels
        self.set_ylabels = set_ylabels
        self.setSquare = setSquare
        self.tightLayout = tightLayout
        self.kwargs0 = kwargs
        self.setUpDims()

        
    def setUpDims(self):
        '''set up the plot'''
        self.fs, self.figsize, self.markersize, self.linewidth = sizes(self.rows, self.cols, self.plotType).values()
        if 'figsize' in self.kwargs0:
            self.figsize = self.kwargs0['figsize']
        plt.rc('font', size=self.fs) 
        self.fig, self.axs = plt.subplots(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, figsize=self.figsize)
        if self.rows==1 or self.cols==1:
            self.axs = np.reshape(self.axs, (self.rows, self.cols))
        self.axlabels = [(i,j) for i in range(self.rows) for j in range(self.cols)]
        self.objs = np.empty((self.rows, self.cols), dtype=object)
        self.ys = np.array([['' for i in range(self.cols)] for j in range(self.rows)], dtype='object')  # array of y variables
        self.xs = np.array([['' for i in range(self.cols)] for j in range(self.rows)], dtype='object')  # array of x variables

        
    def clean(self):
        if self.setSquare:
            for objrow in self.objs:
                for obj in objrow:
                    obj.setSquare()
        if self.plotType=='paper':
            axl = np.array([self.axs[i] for i in self.axlabels])
            subFigureLabels(axl, horiz=True, inside=False)
        if self.set_xlabels:
            for ax in self.fig.axes:
                ax.xaxis.set_tick_params(labelbottom=True)
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
        