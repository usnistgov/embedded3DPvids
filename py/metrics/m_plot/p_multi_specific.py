#!/usr/bin/env python
'''class for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from typing import List, Dict, Tuple, Union, Any, TextIO
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
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

class multiSpecific(multiPlot):
    '''for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvars:np.array, yvars:np.array, cvar:str
                 , mode:str='scatter', yideal:dict={}, legendInAxis:bool=False, holdPlots:bool=False, **kwargs):
        if type(xvars) is list:
            self.xvars = np.array(xvars)
        else:
            self.xvars = xvars
        if type(yvars) is list:
            self.yvars = np.array(yvars)
        else:
            self.yvars = yvars
        self.cvar = cvar
        self.legendMade = legendInAxis
        self.ms = ms
        self.ss = ss.copy()
        self.mode = mode
        self.yideal = yideal
        self.getRC()
        if 'sharex' in kwargs:
            self.sharex = kwargs['sharex']
            kwargs.pop('sharex')
        if 'sharey' in kwargs:
            self.sharey = kwargs['sharey']
            kwargs.pop('sharey')
        if mode=='scatter':
            self.ss.sort_values(by=cvar, inplace=True)
            self.legendVals = self.ss[cvar].unique()
        self.legendInAxis = legendInAxis
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, **kwargs)
        if not holdPlots:
            self.plots()
        
    def getRC(self) -> None:
        '''get rows and columns'''
        self.rows, self.cols = self.yvars.shape
                    
    def getY(self) -> str:
        '''get the y variable'''
        if self.mode=='scatter':
            y = self.yvar
        else:
            y = self.cvar
        return y
    
    def makeLegend(self, i:int, j:int, kwargs:dict) -> None:
        '''turn an axis into a legend'''
        if not self.legendMade:
            self.objs[i,j] = scatterPlot(self.ms, self.ss, justLegend=True, **kwargs)
            self.legendMade = True
            self.axlabels.remove((i,j))
            if i>0:
                self.axs[0,self.cols-1].get_legend().remove()
        else:
            if not self.legendInAxis and not self.legendAbove:
                self.fig.delaxes(self.axs[i,j])
        
    
    def scatterPlot(self, i:int, j:int) -> None:
        '''plot a scatter plot in axis i,j'''
        kwargs = {**self.kwargs0, **{'xvar':self.xvars[i,j], 'cvar':self.cvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType, 'legendVals':self.legendVals}}
        y = self.yvars[i,j]
        if not y in self.ss:
            self.makeLegend(i,j,kwargs)
            return   
        if type(self.yideal) is dict:
            if p in self.yideal:
                kwargs['yideal'] = self.yideal[p]
        else:
            kwargs['yideal'] = self.yideal.yideal(y)
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                 , yvar=y
                                 , set_xlabel=True, set_ylabel=True
                                 , legend=((i==0 and j==self.cols-1) and not self.legendMade)
                                 , legendloc='inset'   
                                 , **kwargs)
        
    def gridPlot(self, func, i:int, j:int) -> None:
        '''plot either a mesh plot or a contour plot on axis i,j'''
        kwargs = {**self.kwargs0, **{'xvar':self.xvars[i,j], 'yvar':self.yvar
                                         , 'ax':self.axs[i,j], 'fig':self.fig
                                         , 'plotType':self.plotType}}
        y = self.yvars[i,j]
        if not y in self.ss:
            self.fig.delaxes(self.axs[i,j])
            return 
        self.ys[i,j]=y
        self.objs[i,j] = func(self.ms, self.ss
                                 , cvar=y
                                 , set_xlabel=(i==self.rows-1), set_ylabel=True
                                 , legend=True
                                 , legendloc='right'
                                 , **kwargs)
    
    def plot(self, i:int, j:int) -> None:
        '''make a single plot on axis i,j'''
        self.xs[i,j] = self.xvars[i,j]
        self.ys[i,j] = self.yvars[i,j]
        if self.mode=='scatter':
            self.scatterPlot(i,j)
        elif self.mode=='mesh':
            self.gridPlot(meshPlot, i,j)
        elif self.mode=='contour':
            self.gridPlot(contourPlot, i,j)

    def plots(self):
        '''plot all plots'''
        if self.legendAbove:
            # put a legend above the axis
            self.legendObj = scatterPlot(self.ms, self.ss, justLegend=True, ax=self.legendAx, fig=self.fig
                                         ,cvar=self.cvar, legendVals=self.legendVals, wideLegend=True, plotType=self.plotType)
            self.legendMade = True
        for i in range(self.rows):
            for j in range(self.cols):
                self.plot(i,j)
        self.clean()
        