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
import tools.regression as rg
from tools.config import cfg
from tools.figureLabels import *
from summary_metric import *
from m_stats import *
from m_plots_color import *
from m_plots_multi import *
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

class yvarlines(multiPlot):
    '''for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvar:str, yvar:str, zvar:str, llist:list=[1,2,3]
                 , plist=['wp', 'wo', 'dwdt', 'wrelax', 'write', 'dp', 'do', 'dddt', 'drelax', 'disturb'],  **kwargs):
        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar
        self.llist = llist
        self.plist = plist
        self.ms = ms
        self.ss = ss.copy()
        self.getRC()
        if 'sharex' in kwargs:
            self.sharex = kwargs['sharex']
            kwargs.pop('sharex')
        else:
            self.sharex = True
        if 'sharey' in kwargs:
            self.sharey = kwargs['sharey']
            kwargs.pop('sharey')
        self.ss.sort_values(by=zvar, inplace=True)
        self.legendVals = self.ss[zvar].unique()
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, **kwargs)
        self.plots()
        
    def getRC(self) -> None:
        '''get rows and columns'''
        self.transpose = False
        self.wdrows = False
        if type(self.plist[0]) is list:
            self.wdrows = True
            # put write and disturb as the rows, only select one line
            if not len(self.llist)==1:
                raise ValueError(f'Too many values in {self.llist}')
            cols = max([len(p) for p in self.plist])
            rows = len(self.plist)
            self.sharey = 'col'
        elif len(self.plist)==1:
            self.transpose = True   # flip rows and columns because we only have one item in llist
            cols = len(self.llist)
            rows = 1
            self.sharey = False
        else:
            cols = len(self.plist)
            rows = len(self.llist)
            self.sharey = 'col'
        self.rows = rows
        self.cols = cols
        
    def getYvar(self, l:int, p:int) -> str:
        '''get the yvar for this l value and p value'''
        if p=='write':
            yvar = f'delta_{self.yvar}_write{l}'
        elif p=='disturb':
            yvar = f'delta_{self.yvar}_disturb{l}'
        elif p[-2:]=='dt':
            ww = p[1]
            yvar = f'd{self.yvar}dt_{ww}{l}o'
        else:
            ww = p[0]
            p1 = p[1:]
            
            if p1=='p':
                yvar = f'{self.yvar}_{ww}{l}p'
            elif p1=='o':
                yvar = f'{self.yvar}_{ww}{l}o'
            elif p1=='relax':
                yvar = f'delta_{self.yvar}_{ww}{l}relax'
            else:
                raise ValueError(f'Could not determine variable for {l}, {p}')
        return yvar

    
    def plot(self, i:int, j:int, y:str, killWritten:bool=True) -> None:
        '''make a single plot'''
        kwargs = {**self.kwargs0, **{'xvar':self.xvar, 'zvar':self.zvar, 'ax':self.axs[i,j], 'fig':self.fig, 'plotType':self.plotType, 'legendVals':self.legendVals}}
        if not y in self.ss:
            if i==0 and j==self.cols-1:
                # turn this into a legend
                self.objs[i,j] = metricPlot(self.ms, self.ss, justLegend=True, **kwargs)
                self.axlabels.remove((i,j))
            else:
                self.fig.delaxes(self.axs[i,j])
            return   
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                     , yvar=y
                                     , set_xlabel=(i==self.rows-1), set_ylabel=False
                                     , legend=(i==0 and j==self.cols-1)
                                     , legendloc='right'
                                     , **kwargs)
        self.columnTitle(i,y,j, killWritten=killWritten)
        
    def columnTitle(self, i:int, y:str, j:int, killWritten:bool=True):
        '''add a title to the plot describing what this column does'''
        if i==0:
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
            self.axs[0,j].set_title(yy, fontsize=self.fs)
            
    def rowTitle(self, i:int, pl:str) -> None:
        if self.plotType=='paper':
            rotation = 90
        else:
            rotation = 0
        self.axs[i,0].set_ylabel(pl, fontsize=self.fs, rotation=rotation)

    def plots(self):
        '''plot all plots'''
        if self.transpose:
            # one row of p values
            l = self.plist[0]
            for i,p in enumerate(self.llist):
                y = self.getYvar(l, p)
                self.plot(0,i,y)
        elif self.wdrows:
            # rows are write and disturb, cols are p values
            l = self.llist[0]
            for i,prow in enumerate(self.plist):
                for j,p in enumerate(prow):
                    y = self.getYvar(l, p)
                    self.plot(i,j,y)
                    
                # row header
                if 'w' in p:
                    pl = 'write'
                else:
                    pl = 'disturb'
                self.rowTitle(i, pl)
                
            # figure title
            yy = self.ms.varSymbol(f'{self.yvar}_w1o').replace('1st written ', '')
            yy = f'Line {l} {yy}'
            self.fig.suptitle(yy)
        else:
            # rows are line number, cols are p values
            write = all(['w' in p for p in self.plist])
            disturb = all(['d' in p for p in self.plist])
            killWritten = write or disturb
            for i,l in enumerate(self.llist):
                for j,p in enumerate(self.plist):
                    y = self.getYvar(l,p)
                    self.plot(i,j,y, killWritten=killWritten)
                self.rowTitle(i, f'Line {i+1}')
                
            # row header
            yy = self.ms.varSymbol(f'{self.yvar}_w1o').replace('1st written ', '')
            if killWritten:
                if write:
                    yy = f'Write {yy}'
                else:
                    yy = f'Disturb {yy}'
            self.fig.suptitle(yy)
        self.clean()