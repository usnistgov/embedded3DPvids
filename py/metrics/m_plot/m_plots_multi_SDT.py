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
from tools.figureLabels import *
from m_summary.summary_metric import *
from m_stats import *
from m_plots_color import *
from m_plots_multi import *
from m_plots_scatter import *
from m_plots_mesh import *
from m_plots_contour import *

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
                 , plist=['wp', 'wo', 'dwdt', 'wrelax', 'write', 'dp', 'do', 'dddt', 'drelax', 'disturb']
                 , mode:str='scatter', yideal:dict={}, **kwargs):
        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar
        self.llist = llist
        self.plist = plist
        self.legendMade = False
        self.ms = ms
        self.ss = ss.copy()
        self.mode = mode
        self.yideal = yideal
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
        self.groupPCols()
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
            if self.mode=='scatter':
                self.sharey = 'col'
            else:
                self.sharey = True
        elif len(self.plist)==1 and not len(self.llist)==1:
            self.transpose = True   # flip rows and columns because we only have one item in llist
            cols = len(self.llist)
            rows = 1
            if self.mode=='scatter':
                self.sharey = False
            else:
                self.sharey = True
        else:
            cols = len(self.plist)
            rows = len(self.llist)
            if self.mode=='scatter':
                self.sharey = 'col'
            else:
                self.sharey = True
        self.rows = rows
        self.cols = cols
        
    def groupPCols(self):
        '''group dependent variable columns by type'''
        if not self.sharey:
            return
        g = [['wp', 'wo', 'dp', 'do'], ['dwdt', 'dddt'], ['wrelax', 'drelax'], ['write', 'disturb']]
        for gi in g:
            lcommon = list(set(gi).intersection(set(self.plist)))  # for each group, find the variables that are in this plot
            if len(lcommon)>1:
                i0 = self.plist.index(lcommon[0])   # get index of first column
                for l in lcommon[1:]:
                    i = self.plist.index(l)   # get index of 2nd column
                    self.axs[0,i].get_shared_y_axes().join(self.axs[0,i0], self.axs[0,i])  # share the y axes for these columns
        
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
    
    def makeLegend(self, i:int, j:int, kwargs:dict) -> None:
        '''turn an axis into a legend'''
        if not self.legendMade:
            self.objs[i,j] = scatterPlot(self.ms, self.ss, justLegend=True, **kwargs)
            self.legendMade = True
            self.axlabels.remove((i,j))
            if i>0:
                self.axs[0,self.cols-1].get_legend().remove()
        else:
            self.fig.delaxes(self.axs[i,j])
        
    
    def scatterPlot(self, i:int, j:int, y:str, p:str, killWritten:bool=True) -> None:
        kwargs = {**self.kwargs0, **{'xvar':self.xvar, 'zvar':self.zvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType, 'legendVals':self.legendVals}}
        if not y in self.ss:
            self.makeLegend(i,j,kwargs)
            return   
        self.ys[i,j]=y
        if type(self.yideal) is dict:
            if p in self.yideal:
                kwargs['yideal'] = self.yideal[p]
        else:
            kwargs['yideal'] = self.yideal.yideal(y)
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                 , yvar=y
                                 , set_xlabel=True, set_ylabel=False
                                 , legend=(i==0 and j==self.cols-1)
                                 , legendloc='right'   
                                 , **kwargs)
        
    def gridPlot(self, func, i:int, j:int, y:str, p:str, killWritten:bool=True) -> None:
        '''either a mesh plot or a contour plot'''
        kwargs = {**self.kwargs0, **{'xvar':self.xvar, 'yvar':self.zvar
                                         , 'ax':self.axs[i,j], 'fig':self.fig
                                         , 'plotType':self.plotType, 'legendVals':self.legendVals}}
        if not y in self.ss:
            self.fig.delaxes(self.axs[i,j])
            return 
        self.ys[i,j]=y
        self.objs[i,j] = func(self.ms, self.ss
                                 , zvar=y
                                 , set_xlabel=(i==self.rows-1), set_ylabel=True
                                 , legend=True
                                 , legendloc='top'
                                 , **kwargs)
    
    def plot(self, i:int, j:int, y:str, p:str, killWritten:bool=True) -> None:
        '''make a single plot'''
        if self.mode=='scatter':
            self.scatterPlot(i,j,y,p,killWritten)
        elif self.mode=='mesh':
            self.gridPlot(meshPlot, i,j,y,p,killWritten)
        elif self.mode=='contour':
            self.gridPlot(contourPlot, i,j,y,p,killWritten)
            
        
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
            
    def setColumnTitle(self, i:int, j:int, yy:str) -> None:
        '''set the column title'''
        self.axs[i,j].set_title(yy, fontsize=self.fs)
        
    def columnTitle(self, i:int, j:int, killWritten:bool):
        '''add a title to the plot describing the y variable in this column'''
        if i>=self.rows or j>=self.cols:
            return
        y = self.ys[i,j]
        yy = self.pTitle(y, killWritten)
        self.setColumnTitle(i, j,yy)
            
    def rowTitle(self, i:int, j:int, pl:str) -> None:
        '''put a title to the left of the row'''
        if i>=self.rows or j>=self.cols:
            return
        if self.mode=='scatter':
            kwargs = {}
            if not self.plotType=='paper':
                kwargs['rotation'] = 0
            self.axs[i,j].set_ylabel(pl, fontsize=self.fs, **kwargs)
            if not self.plotType=='paper':
                self.axs[i,j].yaxis.set_label_coords(-.3, .5)

        else:
            ll = self.axs[i,j].yaxis.get_label().get_text()
            self.axs[i,j].set_ylabel(f'{pl}\n{ll}', fontsize=self.fs)
            
    def firstRow(self, j:int) -> int:
        i = 0
        while i<self.rows and len(self.ys[i,j])==0:
            i = i+1
        return i
    
    def firstCol(self, i:int) -> int:
        j = 0
        while j<self.cols and len(self.ys[i,j])==0:
            j = j+1
        return j
            
    def rowcolTitlesOneRow(self):
        '''add row column, and figure titles when there is only one row of values, with one p value and a range of line numbers'''
        
        # row title
        self.rowTitle(0, self.firstCol(0), self.pTitle(self.ys[i], False))
        
        # column title
        for j,l in enumerate(self.llist):
            self.setColumnTitle(j, f'Line {l}')
                      
        # plot title
        yy = self.ms.varSymbol(self.ys[i])
        self.fig.suptitle(yy)
        
    def rowcolTitlesWDrows(self):
        '''add row and column titles, where rows are write/disturb and cols are variable types'''
        
        # row titles
        for i,prow in enumerate(self.plist):
            if 'w' in prow[0]:
                pl = 'write'
            else:
                pl = 'disturb'
            self.rowTitle(i, self.firstCol(i), pl)
            
        # column titles
        for j,p in enumerate(self.plist[0]):
            self.columnTitle(self.firstRow(j),j, True)
            
        # figure title
        l = self.llist[0]
        yy = self.ms.varSymbol(f'{self.yvar}_w1o').replace('1st written ', '')
        yy = f'Line {l} {yy}'
        self.fig.suptitle(yy)
        
    def rowColTitlesLP(self):
        '''add row and column titles, where rows are line numbers and columns are variable names'''
        write = all(['w' in p for p in self.plist])
        disturb = all(['d' in p for p in self.plist])
        killWritten = write or disturb
        
        # row titles
        for i,l in enumerate(self.llist):
            self.rowTitle(i, self.firstCol(i), f'Line {l}')
            
        # column titles
        for j,p in enumerate(self.plist):
            self.columnTitle(self.firstRow(j),j,killWritten)

        # figure title
        yy = self.ms.varSymbol(f'{self.yvar}_w1o').replace('1st written ', '')
        if killWritten:
            if write:
                yy = f'Write {yy}'
            else:
                yy = f'Disturb {yy}'
        self.fig.suptitle(yy)
            
    def rowcolTitles(self):
        '''add row and column titles'''
        if self.transpose:
            self.rowcolTitlesOneRow()
        elif self.wdrows:
            # rows are write and disturb, cols are p values
            self.rowcolTitlesWDrows()
        else:
            # rows are line number, cols are p values
            self.rowColTitlesLP()
            
    def plotsOneRow(self):
        '''plot values where there's only one row'''
        # one row of p values
        l = self.plist[0]
        for i,p in enumerate(self.llist):
            y = self.getYvar(l, p)
            self.plot(0,i,y, p)
                
    def plotsWDrows(self):
        '''plot values where rows are write and disturb, and columns are y variables'''
        # rows are write and disturb, cols are p values
        l = self.llist[0]
        for i,prow in enumerate(self.plist):
            for j,p in enumerate(prow):
                y = self.getYvar(l, p)
                self.plot(i,j,y, p)
                
    def plotsLP(self):
        '''plot values, where rows are line numbers and columns are variable names'''
        write = all(['w' in p for p in self.plist])
        disturb = all(['d' in p for p in self.plist])
        killWritten = write or disturb
        for i,l in enumerate(self.llist):
            for j,p in enumerate(self.plist):
                y = self.getYvar(l,p)                
                self.plot(i,j,y,p, killWritten=killWritten)

    def plots(self):
        '''plot all plots'''
        if self.transpose:
            self.plotsOneRow()
        elif self.wdrows:
            self.plotsWDrows()
        else:
            self.plotsLP()
        self.rowcolTitles()
        self.clean()