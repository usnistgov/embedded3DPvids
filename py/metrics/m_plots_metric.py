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
from matplotlib.patches import Patch
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

class sizes:
    '''for setting sizes of figures, fonts, and markers'''
    
    def __init__(self, rows:int, cols:int, plotType:str='notebook'):
        self.rows = rows
        self.cols = cols
        self.plotType = plotType
        if self.plotType=='ppt':
            self.fs = 18
            self.getFigSize(14, 7)
            self.markersize=100
            self.linewidth = 2
        elif self.plotType=='paper':
            self.fs = 8
            self.getFigSize(6.5, 8.5)
            self.markersize=20
            self.linewidth = 1
        elif self.plotType=='notebook':
            self.fs = 16
            self.getFigSize(12, 12)
            self.markersize = 40
            self.linewidth = 2
        else:
            raise ValueError(f'Unknown plot type {self.plotType}')
            
    def values(self):
        return self.fs, self.figsize, self.markersize, self.linewidth
            
            
    def getFigSize(self, wmax:float, hmax:float) -> None:
        self.ar = self.rows/self.cols
        wider = [wmax, wmax*self.ar]
        if wider[1]>hmax:
            wider = [w*hmax/wider[1] for w in wider]
        self.figsize = tuple(wider)
        

class metricPlot:
    '''for holding a single plot on a single axis'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame
                 , xvar:str='', yvar:str='', zvar:str=''
                 , plotType:str='ppt'
                 , logx:bool=False, logy:bool=False
                 , set_xlabel:bool=True, set_ylabel:bool=True
                 , axisSymbols:bool=True
                 , dx:float=0.1, dy:float=1
                 , empty:bool=True
                 , cmapname:str='coolwarm'
                 , gradColor:int=colorModes.discreteZvar
                 , justLegend:bool=False
                 ,  **kwargs):
        self.kwargs0 = kwargs
        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar
        self.ms = ms   # dataframe that holds all data
        self.ss = ss.copy()
        self.plotType = plotType
        self.logx = logx
        self.logy = logy
        self.set_xlabel = set_xlabel
        self.set_ylabel = set_ylabel
        self.axisSymbols = axisSymbols
        self.dx = dx
        self.dy = dy
        self.empty = empty
        self.cmapname = cmapname
        self.cmap = cm.get_cmap(self.cmapname)  
        self.gradColor = gradColor
        self.justLegend = justLegend
        self.setUpDims()
        if self.justLegend:
            self.createLegend()
        else:
            self.checkValid()    # check that inputs are valid
            self.setUpAxes()
        
        
    def checkValid(self):
        if not (self.xvar in self.ss and self.yvar in self.ss):
            if not self.xvar in self.ss:
                if not self.yvar in self.ss:
                    raise NameError(f'Variable name {self.xvar} and {self.yvar} are not in table')
                else:
                    raise NameError(f'Variable name {self.xvar} is not in table')
            else:
                raise NameError(f'Variable name {self.yvar} is not in table')
                
    def dropNA(self):
        '''remove NA x values and y values from table'''
        if not self.zvar in self.ss:
    #         logging.warning(f'variable {zvar} is not in table')
            self.gradColor = colorModes.constant
            self.ss = self.ss[[self.xvar,self.yvar]].dropna()  
        else:
            self.ss = self.ss[[self.xvar, self.yvar, self.zvar]].dropna()
        
    def setUpDims(self):
        '''set up the plot'''
        self.fs, self.figsize, self.markersize, self.linewidth = sizes(1,1,self.plotType).values()
        if 'figsize' in self.kwargs0:
            self.figsize = self.kwargs0['figsize']
        if 'fs' in self.kwargs0:
            self.fs = self.kwargs0['fs']
        plt.rc('font', size=self.fs) 
        if 'fig' in self.kwargs0:
            self.fig = self.kwargs0['fig']
            if 'ax' in self.kwargs0:
                self.ax = self.kwargs0['ax']
            else:
                self.ax = self.fig.add_subplot(111)
        else:
            if 'ax' in self.kwargs0:
                self.ax = self.kwargs0['ax']
                self.fig = plt.figure(figsize=self.figsize)
            else:
                self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize)
                
        
            
    #----------------------------------------
        
    def axisLabel(self, axisName:str) -> None:
        '''label the axis. axisName is x or y'''
        var = getattr(self, f'{axisName}var')
        if self.axisSymbols:
            label = self.ms.varSymbol(var, **self.kwargs0)
        else:
            label = var
        units = self.ms.u
        if var in units and len(units[var])>0:
            label = label + f' ({units[var]})'
        getattr(self.ax, f'set_{axisName}label')(label, fontsize=self.fs)
        
    def axisLabels(self) -> None:
        '''get the labels for the x and y axis and label the axis
        xvar and yvar are variable names
        axisSymbols=True to use symbols for axis labels
        set_xlabel and set_ylabel true to label the axes
        '''
        if self.set_xlabel:
            self.axisLabel('x')
        if self.set_ylabel:
            self.axisLabel('y')
        for s in ['x', 'y']:
            getattr(self.ax, f'{s}axis').get_label().set_fontsize(self.fs)
        return
    
    def setLog(self) -> None:
        '''set log scale on axis'''
        if self.logx:
            self.ax.set_xscale('log')
        if self.logy:
            self.ax.set_yscale('log')

        
    def setUpAxes(self):
        '''get figure and axes. xvar and yvar are column names for x and y variables'''
        self.axisLabels()
        self.setLog()
        
    #----------------------------------------

    def setSquare(self):
        '''set the aspect ratio of the axis to square'''
        self.ax.set_box_aspect(1)

    def calcTicks(self, lim:Tuple[float]):
        '''get the major ticks and minor ticks from the limit'''
        ticklims = [np.ceil(np.log10(lim[0])), np.floor(np.log10(lim[1]))]
        tdiff = ticklims[1]-ticklims[0]
        if tdiff<1:
            # range less than 1 order of magnitude
            tm = 0.2
            ticks = []
            while len(ticks)<4:
                ticks = [i*10**ticklims[0] for i in np.arange(tm, 1+tm, tm)]+[i*10**(int(ticklims[1]+1)) for i in np.arange(tm, 1+tm, tm)]
                ticks = [round(i, -int(np.floor(np.log10(tm)))) for i in ticks]
                ticks = [i for i in ticks if (i>lim[0])&(i<lim[1])]
                tm = tm/2
            tminor = 0.1
        elif tdiff<2:
            ticks = [0.5*10**ticklims[0], 10**ticklims[0], 0.5*10**ticklims[1], 10**ticklims[1], 1.5*10**ticklims[1]]
            tminor = 0.1
        elif tdiff<5:
            ticks = [10**i for i in np.arange(ticklims[0], ticklims[1]+1, 1)]
            tminor = 0.1
        else:
            dt = np.ceil(tdiff/4)
            ticks = [10**i for i in np.arange(ticklims[0], ticklims[1]+dt, dt)]
            tminor = 0.5
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(tminor, 1, tminor),numticks=12)
        ticks = [i for i in ticks if (i>lim[0])&(i<lim[1])]
        ticks = list(set(ticks))
        if ticks[0]==0.9 and ticks[-1]>=2:
            ticks = ticks[1:]
        return ticks, locmin

    def fixTicks(self):
        '''fix log scale ticks'''

        if self.logx:
            lim = self.ax.get_xlim()
            ticks, locmin = self.calcTicks(lim)
            self.ax.set_xticks(ticks)
            if min(ticks)>0.1 and max(ticks)<=10:
                self.ax.set_xticklabels(ticks)
            self.ax.xaxis.set_minor_locator(locmin)
            self.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if self.logy:
            lim = self.ax.get_ylim()
            ticks, locmin = self.calcTicks(lim)
            self.ax.set_yticks(ticks)
            if min(ticks)>0.1 and max(ticks)<=10:
                self.ax.set_yticklabels(ticks)
            self.ax.xaxis.set_minor_locator(locmin)
            self.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            
    #----------------------------------------
    
    def onePointSpacing(xl0:list) -> list:
        '''produce a list to encompass a single point'''
        xl0.sort()
        if len(xl0)>1:
            firstpoint = [xl0[0] - (xl0[1]-xl0[0])/2]
            lastpoint = [xl0[-1] + (xl0[-1]-xl0[-2])/2]
            midpoints = [(xl0[i]+xl0[i+1])/2 for i in range(len(xl0)-1)]
            xl = firstpoint + midpoints + lastpoint
        else:
            xl = [xl0[0]-1, xl0[0]+1]
        return xl
    
    def evenlySpace(self, var:str, logv:bool, dv:float) -> list:
        '''produce an evenly spaced list
        var is the x or y variable
        logv True to evenly space on a log scale
        dv is the spacing between points, as a fraction of the total range'''
        s3 = self.ss[var].dropna()
        if logv:
            # convert to log scale. only take positive values
            s3 = s3[s3>0]
            if len(s3)==0:
                # no points left, return
                return []
        if (dv>0 and dv<1) and len(s3.unique())>1:
            # more than 1 point along this axis
            if logv:
                # convert to log
                logs = np.log10(s3)
                vmin = min(logs)     # find min and max values of log scale variables
                vmax = max(logs)
                ddv = (vmax-vmin)*dv # spacing between values 
                xl = [10**i for i in np.arange(vmin-ddv*0.5,vmax+ddv,ddv)]   # list of values to cut between
            else:
                # linear scale
                vmin = s3.min()    # find min and max
                vmax = s3.max()
                ddv = (vmax-vmin)*dv   # spacing between values
                xl = list(np.arange(vmin-ddv*0.5,vmax+ddv,ddv))   # list of values to cut between
        elif dv>=1:
            xl = [s3.min()-1, s3.max()+1]    # requested spacing is over full range. don't split this set at all
        else:
            xl0 = list(s3.unique())           # only one value in this set
            return self.onePointSpacing(xl0)
        xl.sort()
        return xl
    
    def gridValue(self, ss3:pd.DataFrame, rigid:bool, x:float, y:float, xl:list, yl:list, i:int, j:int) -> dict:
        '''get a single x,y,color value for the grid'''
        if rigid:
            # use the center of the square
            if self.logx:
                # put middles on log scale
                xm = 10**((np.log10(x)+np.log10(xl[i+1]))/2)
            else:
                # put middles on linear scale
                xm = (x+xl[i+1])/2
            if self.logy:
                ym = 10**((np.log10(y)+np.log10(yl[j+1]))/2)
            else:
                ym = (y+yl[j+1])/2
            xerr = 0
            yerr = 0
        else:
            # measure mean and standard error
            xm, xerr, _ = pooledSEDF(ss3, self.xvar)
            ym, yerr, _ = pooledSEDF(ss3, self.yvar)
        if self.zvar in ss3:
            # get color
            try:
                c0, _, _ = pooledSEDF(ss3, self.zvar)
            except:
                c0 = 0
        else:
            c0 = 0
        return {'x':xm, 'xerr':xerr, 'y':ym, 'yerr':yerr, 'c':c0}
    
    def toGrid(self, ss2:pd.DataFrame, rigid:bool=False) -> pd.DataFrame:
        '''convert the data to an evenly spaced grid. 
        xvar, yvar, zvar are variable names (columns in ss2)
        logx and logy True to space variables on log scale
        dx and dy are spacing as a fraction of the total range
        rigid=True means that we use the center of the square as the x,y value and use no error. This is for color maps.'''
        xl = self.evenlySpace(self.xvar, self.logx, self.dx)
        if len(xl)==0:
            return []
        yl = self.evenlySpace(self.yvar, self.logy, self.dy)
        df2 = []
        for j,y in enumerate(yl[:-1]):
            for i,x in enumerate(xl[:-1]):
                ss3 = ss2.copy()
                if self.logx:
                    ss3 = ss3[ss3[self.xvar]>0]
                if self.dx<1:
                    ss3 = ss3[(ss3[self.xvar]>=x)&(ss3[self.xvar]<xl[i+1])]
                if self.dy<1:
                    ss3 = ss3[(ss3[self.yvar]>=y)&(ss3[self.yvar]<yl[j+1])] # points within this grid square
                if len(ss3)>0 or rigid:
                    df2.append(self.gridValue(ss3, rigid, x, y, xl, yl, i, j))
        df2 = pd.DataFrame(df2)
        return df2

    def toGroups(self, ss2:pd.DataFrame) -> pd.DataFrame:
        '''group the data into groups of equal size and get errors
        xvar, yvar, zvar are variable names (columns in ss2)
        logx and logy True to space variables on log scale
        dx and dy are spacing as a fraction of the total range'''
        if not ((self.dx==1) or (self.dy==1)):
            # x or y need to be averaged for groups to work. if both are split, use a grid
            return self.toGrid()
        if self.dy==1:
            ss2.sort_values(by=self.xvar, inplace=True) # sort the dataframe by x values
            active = self.dx
        else:
            ss2.sort_values(by=self.yvar, inplace=True) # sort the dataframe by x values
            active = self.dy
        if active>0:
            # find the number of items per group
            pergrp = int(np.ceil(len(ss2)*active)) # items per group
            numgrps = int(np.ceil(1/active))
        else:
            pergrp = 1
            numgrps = len(ss2)
        df2 = []
        for i in range(numgrps):
            ss3 = ss2.iloc[((i-1)*pergrp):min(len(ss2),(i*pergrp))]
            xm, xerr, _ = pooledSEDF(ss3, self.xvar)
            ym, yerr, _ = pooledSEDF(ss3, self.yvar)
            if self.zvar in ss3:
                try:
                    c0, _, _ = pooledSEDF(ss3, self.zvar)
                except:
                    c0 = 0
            else:
                c0 = 0
            df2.append({'x':xm, 'xerr':xerr, 'y':ym, 'yerr':yerr, 'c':c0})
        df2 = pd.DataFrame(df2)
        return df2
    
    #--------------------------------------------------------------
    
    def getColor(self, i:int, legendVals:list) -> str:
        '''get the color for a single series'''
        if 'color' in self.kwargs0:
            # same color, vary marker
            return self.kwargs0['color']
        elif 'edgecolors' in self.kwargs0 and not self.kwargs0['edgecolors']=='none':
            return self.kwargs0['edgecolors']
        else:
            # different color, same marker
            if len(legendVals)>1:
                ifrac = (i)/(len(legendVals)-1)
                color = self.cmap(ifrac)
                if (ifrac>0.35 and ifrac<0.65) and self.cmapname=='coolwarm':
                    if ifrac==0.5:
                        return 'gray'
                    else:
                        return adjust_saturation(adjust_lightness(color, 0.9), 0.5) # darken and desaturate middle color
                else:
                    return color
            else:
                return 'black'
            
    def getMarker(self, i:int, legendVals:list, color:str) ->  dict:
        '''get the marker for a single series'''
        varargs = {}
        if 'marker' in self.kwargs0:
            varargs['marker'] = self.kwargs0['marker']
        else:
            mlist = ['o','v','s','^','X','D','v']
            varargs['marker'] = mlist[i%len(mlist)]
        
        if self.empty:
            varargs['facecolors']='none'
            varargs['edgecolors']=color
            varargs['color']=color 
        else:
            varargs['color']=color  
            varargs['edgecolors']='none'
            varargs['facecolors']=color
            
        if hasattr(self, 'linewidth'):
            varargs['linewidth'] = self.linewidth
        else:
            linewidth = 2
            
        if hasattr(self, 'markersize'):
            varargs['s'] = self.markersize
        
        for sname in ['facecolors', 'edgecolors', 'markersize']:
            if sname in self.kwargs0:
                varargs[sname] = self.kwargs0[sname]
        return varargs
    
    def seriesColor(self, i:int, legendVals:List) -> dict:
        '''get the color for this series and put it in the plot args dictionary. 
        gradColor=1 to color by a gradient color, otherwise group by a variable and choose colors
        cmapname is the colormap name, for matplotlib colormaps
        i is the current index in legendVals
        legendVals is a list of values in the legend
        empty=True to leave the center of the marker empty
        '''
        if self.gradColor==colorModes.discreteZvar or self.gradColor==colorModes.constant:
            color = self.getColor(i, legendVals)
            varargs = self.getMarker(i, legendVals, color)
        elif self.gradColor==colorModes.gradientZvar:
            # color by gradient
            varargs = {}
        else:
            raise ValueError(f'Unexpected gradColor value {self.gradColor}')
        return varargs
    
    def createLegend(self):
        '''this axis just going to be a legend'''
        if 'legendVals' in self.kwargs0:
            self.legendVals = self.kwargs0['legendVals']
        else:
            raise ValueError('Need input legendVals to create legend')
            
        # create handles with no point plotted but styles added
        for i,l in enumerate(self.legendVals):
            color = self.getColor(i, self.legendVals)
            varargs = self.getMarker(i, self.legendVals, color)
            self.ax.scatter([],[], label=l, **varargs)
        
        l = self.ax.legend(title=self.ms.varSymbol(self.zvar), loc="center", fontsize=self.fs, framealpha=1, frameon=False)
        self.ax.axis('off')
    
    #------------------------------------------
    
    def idealLines(self) -> None:
        '''add vertical and horizontal lines'''
        varargs = {'color':'gray', 'linestyle':'--', 'linewidth':'1', 'label':'ideal'}
        if 'xideal' in self.kwargs0:
            self.ax.axvline(self.kwargs0['xideal'], 0,1, **varargs)
            varargs.pop('label')
        if 'yideal' in self.kwargs0:
            self.ax.axhline(self.kwargs0['yideal'], 0,1, **varargs)
        