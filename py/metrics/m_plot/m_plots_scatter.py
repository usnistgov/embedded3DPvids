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
import warnings
warnings.simplefilter('error', UserWarning)

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

class scatterPlot(metricPlot):
    '''scatter plot of measured values. 
        xvar is the x variable name, yvar is the y variable name. 
        zvar is the variable to color by. 
        logx, logy to plot on a log scale
        
        dx>0 to group points by x and take error. otherwise, plot everything. dx=1 to average all points together
        cmapname is the name of the colormap in matplotlib
        plotReg to plot linear regression on top of the plot
        grid=True to group the points into equal spacings. grid=False to group the points into equal numbers of points
        lines=True to plot connecting lines between points
        '''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, plotReg:bool=False
                 , grid:bool=True, lines:bool=False, dx:float=0.1, dy:float=1, **kwargs):
        self.plotReg = plotReg
        self.grid = grid
        self.lines = lines
        super().__init__(ms, ss, dx=dx, dy=dy, **kwargs)
        if not self.justLegend:
            self.plot()
            
    def getLegendVals(self) -> None:
        '''get a list of values to use to split into series. if gradient color scheme, all one series'''
        if 'legendVals' in self.kwargs0:
            self.legendVals = self.kwargs0['legendVals']
        if self.gradColor==colorModes.discreteZvar:
            self.legendVals = list(self.ss[self.zvar].unique())
            self.legendVals.sort()
        else:
            self.legendVals = [0]
            
    def plotSeriesGradient(self, df2:pd.DataFrame, varargs:dict) -> None:
        '''plot the series with gradient color based on zvar'''
        # plot the points
        self.sc = self.ax.scatter(df2['x'], df2['y']
                                  ,linestyle='None', zorder=100
                                  ,c=df2['c'], cmap=self.cmapname
                                  ,**varargs)

        # plot error bars
        for s in ['label', 'linewidth']:
            if s in varargs:
                if s=='linewidth':
                    varargs['elinewidth'] = varargs.pop(s)
                else:
                    varargs.pop(s)
        cmin = df2.c.min()
        cmax = df2.c.max()
        if self.dx>0 and self.dy>0: 
            for j, row in df2.iterrows():
                color = self.cmap((row['c']-cmin)/(cmax-cmin))
                self.sc = self.ax.errorbar([row['x']],[row['y']]
                                           , xerr=[row['xerr']], yerr=[row['yerr']]
                                           , linestyle='None', color=color,**varargs)
                
    def plotSeriesConstant(self, df2:pd.DataFrame, varargs:dict) -> None:
        '''plot series, with constant color'''
        # plot points
        self.sc = self.ax.scatter(df2['x'],df2['y'], **varargs)
        
        # plot error bar
        for s in ['label', 'facecolors', 'edgecolors', 's', 'fillstyle', 'marker', 'linewidth']:
            if s in varargs:
                if s=='linewidth':
                    varargs['elinewidth'] = varargs.pop(s)
                else:
                    varargs.pop(s)
        try:
            self.sc = self.ax.errorbar(df2['x'],df2['y'], xerr=df2['xerr']
                              , yerr=df2['yerr'],linestyle='None'
                              , marker='', **varargs)
        except Exception as e:
            print(df2)
            raise e
            
    def plotSeries(self, df2:pd.DataFrame, varargs:dict) -> None:
        '''df2 is already sorted into x,y,c where c is color
        gradColor=1 to use gradient coloring as a function of c, otherwise color by group
        ax is the axis to plot on
        cmapname is the name of the colormap in matplotlib
        dx and dy are the spacing between points, as a fraction of the total range
        lines=True to plot connecting lines between points
        '''
        if len(df2)==0:
            self.sc = self.ax.scatter(np.NaN, np.NaN, **varargs)
            return
        if self.gradColor==colorModes.gradientZvar:
            # gradient coloring
            self.plotSeriesGradient(df2, varargs)            
        else:
            # fixed color
            self.plotSeriesConstant(df2, varargs)
            
        if self.lines:
            # plot lines between points
            self.sc = self.ax.plot(df2['x'], df2['y'], **varargs)            
            
    def plotGroup(self, i:int, val:Any) -> None:
        '''get plot settings and plot the group'''
        varargs = self.seriesStyle(i, self.legendVals)
        if self.gradColor==colorModes.discreteZvar:
            # split series by zvar
            ss2 = self.ss[self.ss[self.zvar]==val]
            varargs['label']=val
        else:
            # plot all in one group
            ss2 = self.ss
            if 'label' in self.kwargs0:
                varargs['label'] = self.kwargs0['label']

        if len(ss2)>0:
            if self.grid:
                # evenly space groups
                df2 = self.toGrid(ss2)
            else:
                # even population in groups
                df2 = self.toGroups(ss2)
            self.plotSeries(df2, varargs)
            
    def addLegends(self):
        '''add legend to the plot'''
        
        # no legend
        if ('legend' in self.kwargs0 and self.kwargs0['legend']==False):
            return
        
        # color bar legend
        zlabel = self.zvar.replace('_',  ' ')
        if self.gradColor==colorModes.gradientZvar and self.zvar in self.ss:
            self.cbar = plt.colorbar(self.sc, label=zlabel)
            return
        
        # swatch legend
        handles, labels = self.ax.get_legend_handles_labels()
        if len(labels)==0:
            return
        
        if 'legendloc' in self.kwargs0:
            legendloc = self.kwargs0['legendloc']
        else:
            legendloc = ''        
        
        if len(self.fig.axes)==1 or (legendloc=='right'):
            self.ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', title=zlabel, frameon=False)
        elif (legendloc=='above'):
            self.ax.legend(bbox_to_anchor=(0,1), loc='lower left', title=zlabel, frameon=False)
        elif (legendloc=='below'):
            self.ax.legend(bbox_to_anchor=(0,-0.5), loc='upper left', title=zlabel, frameon=False)
        elif (legendloc=='inset'):
            self.ax.legend(bbox_to_anchor=(1,0), loc='lower right', frameon=True)
        else:
            self.ax.legend(bbox_to_anchor=(0,1), loc='lower left', title=zlabel, frameon=False)
            
    def regressionSS(self) -> None:
        '''add a linear regression to the plot'''
        if self.logx:
            self.xvar = self.xvar+'_log'
            if not self.xvar in self.ss:
                self.ss = self.ms.addLogs(self.ss, [self.xvar[:-4]])
        if self.logy:
            self.yvar = self.yvar+'_log'
            if not self.yvar in self.ss:
                self.ss = self.ms.addLogs(self.ss, [self.yvar[:-4]])
        ss2 = self.ss.copy()
        ss2.replace([np.inf, -np.inf], np.nan, inplace=True)  # remove infinite values
        ss2 = ss2.dropna(subset=[self.xvar,self.yvar])
        reg = rg.regPD(ss2, [self.xvar], self.yvar)
        min1 = ss2[self.xvar].min()
        max1 = ss2[self.xvar].max()
        logxlist = list(np.arange(min1, max1, (max1-min1)/20))
        ylist = [reg['c']+reg['b']*x for x in logxlist]
        if xvar[-3:]=='log':
            xlist = [10**i for i in logxlist]
        else:
            xlist = logxlist
        if yvar[-3:]=='log':
            ylist = [10**i for i in ylist]
        ylim = self.ax.get_ylim()
        xlist = [xlist[i] for i, y in enumerate(ylist) if ((y>ylim[0])&(y<ylim[1]))]
        ylist = [ylist[i] for i, y in enumerate(ylist) if ((y>ylim[0])&(y<ylim[1]))]
        self.ax.plot(xlist, ylist, color='black')
        self.ax.text(0.9, 0.9, 'r$^2$ = {:0.2}'.format(reg['r2']), transform=self.ax.transAxes)
            
        
    def plot(self):
        '''make the plot'''
        self.dropNA()   # drop nan values
        self.getLegendVals()   # get list of groups
        
        # iterate through list of series
        for i,val in enumerate(self.legendVals):
            self.plotGroup(i,val)

        # add verticals, horizontals
        self.idealLines()

        # add legends
        self.addLegends()      

        # set square
        self.setSquare()
        self.fixTicks()
        if self.plotReg:
            self.regressionSS()
            
    def createLegend(self):
        '''this axis just going to be a legend'''
        if 'legendVals' in self.kwargs0:
            self.legendVals = self.kwargs0['legendVals']
        else:
            raise ValueError('Need input legendVals to create legend')
            
        # create handles with no point plotted but styles added
        for i,l in enumerate(self.legendVals):
            color = self.getColor(i, self.legendVals)
            varargs = self.getMarker(i, color)
            self.ax.scatter([],[], label=l, **varargs)
        
        l = self.ax.legend(title=self.ms.varSymbol(self.zvar), loc="center", fontsize=self.fs, framealpha=1, frameon=False)
        self.ax.axis('off')
