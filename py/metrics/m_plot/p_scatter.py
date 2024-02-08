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
from p_metric import metricPlot
from legend import plotLegend
from markers import plotMarkers

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
    
    def __init__(self, ms, ss:pd.DataFrame, plotReg:bool=False
                 , grid:bool=True, lines:bool=False, dx:float=0.1, dy:float=1
                 , wideLegend:bool=False, **kwargs):
        self.plotReg = plotReg
        self.grid = grid
        self.lines = lines
        super().__init__(ms, ss, dx=dx, dy=dy, **kwargs)
        self.getMarkers()
        if not self.justLegend:
            self.plot()
        else:
            self.createLegend(wideLegend=wideLegend)
            
    def getMarkers(self):
        '''get the marker styles'''
        if self.mvar in self.ss:
            vallist = list(self.ss[self.mvar].unique())
        else:
            vallist = []
        self.markers = plotMarkers(vallist, self.mvar, self.ms.varSymbol(self.mvar)
                                   , self.sizes.markersize, self.sizes.linewidth, lines=self.lines, **self.kwargs0)
            
    def plotSeriesGradient(self, df2:pd.DataFrame, varargs:dict) -> None:
        '''plot the series with gradient color based on zvar'''
        # plot the points
        self.sc = self.ax.scatter(df2['x'], df2['y']
                                  ,linestyle='None', zorder=100
                                  ,c=df2['c'], cmap=self.cmapname
                                  ,**varargs)
        if not self.errorBars:
            return
        if not 'xerr' in df2:
            return
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
        if not self.errorBars:
            return
        if not 'xerr' in df2:
            return
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
        self.plotSeriesConstant(df2, varargs)
            
        if self.lines:
            # plot lines between points
            self.sc = self.ax.plot(df2['x'], df2['y'], **varargs)            
            
    def plotGroup(self, ssi:pd.DataFrame, cvar:Any, mvar:Any, label:str) -> None:
        '''get plot settings and plot the group'''
        if len(ssi)==0:
            return
        color = self.colors.getColor(cvar)
        style = self.markers.markerStyle(mvar, color)
        if len(str(label))>0:
            style['label'] = label

        if self.grid:
            # evenly space groups and get error bars
            df2 = self.toGrid(ssi, **self.kwargs0)
        else:
            # even population in groups and get error bars
            df2 = self.toGroups(ssi, **self.kwargs0)
        self.plotSeries(df2, style)
        
    def addLegend(self, wideLegend:bool=False, **kwargs) -> None:
        '''add the legend to the plot'''
        self.legend = plotLegend(self.colors, self.markers, self.ms, self.ss, self.lines, fs=self.fs, wideLegend=wideLegend, **self.kwargs0)
        
        # color bar legend
        if not self.colors.swatch and len(self.colors.vallist) and self.cvar in self.ss:
            self.cbar = self.legend.colorBar(self.fig, **self.kwargs0)
            return
        
        # swatch legend
        self.legend.swatchLegend(self.fig, self.ax, **kwargs, **self.kwargs0)
            
    def addLegends(self):
        '''add legend to the plot'''
        # no legend
        if ('legend' in self.kwargs0 and self.kwargs0['legend']==False):
            return
        self.addLegend()
        
    def createLegend(self, wideLegend:bool=False):
        '''this axis just going to be a legend'''
        self.addLegend(legendloc='center', wideLegend=wideLegend)
        self.ax.axis('off')
            
            
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
        # add legends
        self.addLegends()     
        # iterate through list of series
        if len(self.colors.vallist)>0:
            # discrete colors
            for cval in self.colors.vallist:
                ssi = self.ss[self.ss[self.cvar]==cval]
                if self.mvar==self.cvar:
                    self.plotGroup(ssi, cval, cval, cval)
                elif len(self.markers.vallist)>0 and not self.mvar==self.cvar:
                    # discrete markers
                    for mval in self.markers.vallist:
                        ssij = ssi[ssi[self.mvar]==mval]
                        self.plotGroup(ssij, cval, mval, f'{cval}, {mval}')
                else:
                    self.plotGroup(ssi, cval, '', cval)
        else:
            if len(self.markers.vallist)>0:
                # discrete markers
                for mval in self.markers.vallist:
                    ssj = ss[ss[self.mvar]==mval]
                    self.plotGroup(ssij, '', mval, mval)
            else:
                self.plotGroup(self.ss, '', '', '')

        # add verticals, horizontals
        self.idealLines()

        # set square
        self.setSquare()
        self.fixTicks()
        if self.plotReg:
            self.regressionSS()

