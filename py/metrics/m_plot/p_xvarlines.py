#!/usr/bin/env python
'''Functions for plotting still and video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from m_summary.summary_metric import summaryMetric
from m_stats import *
from p_multi import multiPlot
from p_scatter import scatterPlot
from p_mesh import meshPlot
from p_contour import contourPlot

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

class xvarlines(multiPlot):
    '''for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvarList:List[str], yvar:str, cvar:str
                 , mode:str='scatter', yideal:dict={}, cols:int=4, **kwargs):
        self.xvarList = xvarList
        self.yvar = yvar
        self.cvar = cvar
        self.legendMade = False
        self.ms = ms
        self.ss = ss.copy()
        self.mode = mode
        self.yideal = yideal
        self.cols = min(4, len(self.xvarList))
        self.rows = int(np.ceil(len(self.xvarList)/self.cols))
        self.sharex = False
        self.sharey = True
        self.ss.sort_values(by=cvar, inplace=True)
        self.legendVals = self.ss[cvar].unique()
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, **kwargs)
        self.plots()

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
        
    
    def scatterPlot(self, i:int, j:int, x:str) -> None:
        '''plot a scatter plot'''
        kwargs = {**self.kwargs0, **{'xvar':x, 'cvar':self.cvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType, 'legendVals':self.legendVals}}
        y = self.yvar
        if not y in self.ss:
            self.makeLegend(i,j,kwargs)
            return   
        
        if type(self.yideal) is dict:
            if y in self.yideal:
                kwargs['yideal'] = self.yideal[y]
        else:
            kwargs['yideal'] = self.yideal.yideal(y)
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                 , yvar=self.yvar
                                 , set_xlabel=True, set_ylabel=False
                                 , legend=(i==0 and j==self.cols-1)
                                 , legendloc='right'   
                                 , **kwargs)
        
    def gridPlot(self, func, i:int, j:int, x:str) -> None:
        '''either a mesh plot or a contour plot'''
        kwargs = {**self.kwargs0, **{'xvar':x, 'yvar':self.zvar
                                         , 'ax':self.axs[i,j], 'fig':self.fig
                                         , 'plotType':self.plotType, 'legendVals':self.legendVals}}
        if not y in self.ss:
            self.fig.delaxes(self.axs[i,j])
            return 
        self.objs[i,j] = func(self.ms, self.ss
                                 , zvar=self.yvar
                                 , set_xlabel=True, set_ylabel=True
                                 , legend=True
                                 , legendloc='top'
                                 , **kwargs)
        
    def labelAx(self, i:int, label:str) -> None:
        '''give the axis a supfigure label'''
        self.axs[int(np.floor(i/self.cols)), i%self.cols].set_title(label, fontsize=self.objs[0,0].fs)
    
    def plot(self, i:int, j:int, x:str) -> None:
        '''make a single plot'''
        if self.mode=='scatter':
            self.scatterPlot(i,j,x)
        elif self.mode=='mesh':
            self.gridPlot(meshPlot, i,j,x)
        elif self.mode=='contour':
            self.gridPlot(contourPlot, i,j,x)
            
    def plots(self) -> None:
        '''make all of the plots'''
        for i in range(self.rows):
            for j in range(self.cols):
                n = i*self.cols+j
                if n<len(self.xvarList):
                    x = self.xvarList[n]
                    self.xs[i,j]=x
                    self.plot(i,j,x)
                else:
                    self.fig.delaxes(self.axs[i,j])
        for row in range(self.rows):
            self.objs[row, 0].axisLabel('y')
        self.clean()
        