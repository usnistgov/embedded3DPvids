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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import regression as rg
import metrics as me
from config import cfg



# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"


#-------------------------------------------------------------

def tossBigSE(df:pd.DataFrame, column:str, quantile:float=0.9):
    '''toss big standard errors from the list'''
    if not column[-3:]=='_SE':
        column = column+'_SE'
    return df[df[column]<df[column].quantile(quantile)]

def cubehelix1(val:float):
    '''val should be 0-1. returns a color'''
    cm = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
    return cm(val)


def adjust_lightness(color, amount=0.5):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_saturation(color, amount=0.5):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1], max(0, min(1, amount * c[1])))

def roundToOrder(val:float):
    if abs(val)<10**-14:
        return 0
    else:
        return round(val, -int(np.log10(abs(val)))+1)
    
def simplifyType(s:Union[str, pd.DataFrame]):
    '''if given a dataframe, simplify the sweepType. otherwise, simplify the individual string'''
    if type(s) is str:
        # just take first 2 elements
        spl = re.split('_', s)
        return spl[0]+'_'+spl[1]
    else:
        # convert all strings in sweepType column
        s.loc[:,'sweepType'] = [simplifyType(si) for si in s['sweepType']]


def pooledSE(df:pd.DataFrame, var:str) -> None:
    if 'xs' in var:
        n = 4
    elif 'vert' in var:
        n = 4
    elif 'horiz' in var:
        n = 3
    mean = df[var].mean()
    sevar = var+'_SE'
    if len(df)>1:
        if sevar in df:
            a = np.sum([n*(np.sqrt(n)*row[sevar])**2 for i,row in df.iterrows()])/(n*len(df))
            b = np.sum([n**2*(df.iloc[i][var]-df.iloc[i+1][var])**2 for i in range(len(df)-1)])/(n**2*len(df))
            poolsd = np.sqrt(a+b)
            se = poolsd/np.sqrt(len(df))
        else:
            se = df[var].sem()
    else:
        se = 0
    return mean, se


def onePointSpacing(xl0:list) -> list:
    '''produce a list to encompass a single point'''
    xl0.sort()
    if len(xl0)>1:
        xl = [xl0[i]-0.49*(xl0[i+1]-xl0[i]) for i in range(len(xl0)-1)] + [xl0[-1]+i*(xl0[-1]-xl0[-2]) for i in [-0.49, 0.49]]
    else:
        xl = [xl0[0]-1, xl0[0]+1]
    return xl

def evenlySpace(ss2:pd.DataFrame, xvar:str, logx:bool, dx:float) -> list:
    '''produce an evenly spaced list'''
    s3 = ss2[xvar].dropna()
    if (dx>0 and dx<1) and len(s3.unique())>1:
        if logx:
            s3 = s3[s3>0]
            if len(s3)==0:
                return []
            elif len(s3.unique())==1:
                return onePointSpacing(list(s3.unique()))
            logs = np.log10(s3)
            xmin = min(logs)
            xmax = max(logs)
            ddx = (xmax-xmin)*dx
            if ddx==0:
                xl0 = list(s3.unique())
                return onePointSpacing(xl0)
            xl = [10**i for i in np.arange(xmin-ddx*0.5,xmax+ddx*0.51,ddx)]
        else:
            xmin = s3.min()
            xmax = s3.max()
            ddx = (xmax-xmin)*dx
            xl = list(np.arange(xmin-ddx*0.5,xmax+ddx*0.51,ddx))
    elif dx>=1:
        xl = [s3.min()-1, s3.max()+1]
    else:
        xl0 = list(s3.unique())
        return onePointSpacing(xl0)
    return xl

def toGrid(ss2:pd.DataFrame, xvar:str, yvar:str, zvar:str, logx:bool, logy:bool, dx:float, dy:float, rigid:bool=False) -> pd.DataFrame:
    '''convert the data to an evenly spaced grid. 
    rigid=True means that we use the center of the square as the x,y value and use no error. This is for color maps.'''
    xl = evenlySpace(ss2, xvar, logx, dx)
    yl = evenlySpace(ss2, yvar, logy, dy)
    df2 = []
    for j,y in enumerate(yl[:-1]):
        for i,x in enumerate(xl[:-1]):
            ss3 = ss2.copy()
            if dx<1:
                ss3 = ss3[(ss3[xvar]>=x)&(ss3[xvar]<xl[i+1])]
            if dy<1:
                ss3 = ss3[(ss3[yvar]>=y)&(ss3[yvar]<yl[j+1])] # points within this grid square
            if len(ss3)>0 or rigid:
                if rigid:
                    if logx:
                        xm = 10**((np.log10(x)+np.log10(xl[i+1]))/2)
                    else:
                        xm = (x+xl[i+1])/2
                    if logy:
                        ym = 10**((np.log10(y)+np.log10(yl[j+1]))/2)
                    else:
                        ym = (y+yl[j+1])/2
                    xerr = 0
                    yerr = 0
                else:
                    xm, xerr = pooledSE(ss3, xvar)
                    ym, yerr = pooledSE(ss3, yvar)
                if zvar in ss3:
                    try:
                        c0, _ = pooledSE(ss3, zvar)
                    except:
                        c0 = 0
                else:
                    c0 = 0
                df2.append({'x':xm, 'xerr':xerr, 'y':ym, 'yerr':yerr, 'c':c0})
    df2 = pd.DataFrame(df2)
    return df2

def toGroups(ss:pd.DataFrame, xvar:str, yvar:str, zvar:str, logx:bool, logy:bool, dx:float, dy:float) -> pd.DataFrame:
    '''group the data into groups of equal size and get errors'''
    if not ((dx==1) or (dy==1)):
        # x or y need to be averaged for groups to work. if both are split, use a grid
        return toGrid(ss, xvar, yvar, zvar, logx, logy, dx, dy)
    ss2 = ss.copy()
    if dy==1:
        ss2.sort_values(by=xvar, inplace=True) # sort the dataframe by x values
        active = dx
    else:
        ss2.sort_values(by=yvar, inplace=True) # sort the dataframe by x values
        active = dy
    if active>0:
        pergrp = int(np.ceil(len(ss2)*active)) # items per group
        numgrps = int(np.ceil(1/active))
    else:
        pergrp = 1
        numgrps = len(ss2)
    df2 = []
    for i in range(numgrps):
        ss3 = ss2.iloc[((i-1)*pergrp):min(len(ss2),(i*pergrp))]
        xm, xerr = pooledSE(ss3, xvar)
        ym, yerr = pooledSE(ss3, yvar)
        if zvar in ss3:
            try:
                c0, _ = pooledSE(ss3, zvar)
            except:
                c0 = 0
        else:
            c0 = 0
        df2.append({'x':xm, 'xerr':xerr, 'y':ym, 'yerr':yerr, 'c':c0})
    df2 = pd.DataFrame(df2)
    return df2


def setUpAxes(xvar:str, yvar:str, **kwargs):
    '''get figure and axes'''
    if 'fig' in kwargs:
        fig = kwargs['fig']
    else:
        fig = plt.figure()
        kwargs['fig'] = fig
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(111)
        kwargs['ax'] = ax
    axisLabels(xvar, yvar, **kwargs)
    return fig, ax, kwargs

def axisLabels(xvar:str, yvar:str, **kwargs) -> None:
    '''get the labels for the x and y axis and label the axis'''
    xlabel = xvar
    ylabel = yvar
    if 'units' in kwargs:
        units = kwargs['units']
        if xvar in units and len(units[xvar])>0:
            xlabel = xlabel + f' ({units[xvar]})'
        if yvar in units and len(units[yvar])>0:
            ylabel = ylabel + f' ({units[yvar]})'
    kwargs['ax'].set_xlabel(xlabel)
    kwargs['ax'].set_ylabel(ylabel)
    return

def setLog(ax, logx:bool, logy:bool) -> None:
    '''set log scale on axis'''
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
        
def getMarker(i:int, color) -> dict:
    '''get marker parameters for an index'''
    varargs = {}
    if i==1:
#         varargs['marker'] = ''
        varargs['facecolors']='none'
        varargs['edgecolors']=color
        varargs['color']=color 
    else:
        mlist = ['o','o','v','s','^','X','D','v']
        varargs['marker'] = mlist[i%len(mlist)]
        varargs['color']=color  
        varargs['edgecolors']='none'
        varargs['facecolors']=color

    return varargs
        
def seriesColor(gradColor:bool, cmapname:str, i:int, lst:List, **kwargs) -> dict:
    '''get the color for this series and put it in the plot args dictionary. i should be between 0 and 1, indicating color'''
    if gradColor==0 or gradColor==2:

        varargs = {}
        if 'color' in kwargs:
            # same color, vary marker
            color = kwargs['color']
        elif 'edgecolors' in kwargs and not kwargs['edgecolors']=='none':
            color = kwargs['edgecolors']
        else:
            # different color, same marker
            if len(lst)>1:
                ifrac = (i)/(len(lst)-1)
                cmap = cm.get_cmap(cmapname)   
                color = cmap(ifrac)
                if (ifrac>0.35 and ifrac<0.65) and cmapname=='coolwarm':
                    if ifrac==0.5:
                        color='gray'
                    else:
                        color = adjust_saturation(adjust_lightness(color, 0.9), 0.5) # darken and desaturate middle color
            else:
                color = 'black'
        if 'marker' in kwargs:
            if kwargs['marker']==1:
                varargs = getMarker(1, color)
            else:
                varargs['marker'] = kwargs['marker']
        elif not 'edgecolors' in kwargs:
            varargs = getMarker(i, color)
        for sname in ['facecolors', 'color', 'edgecolors']:
            if sname in kwargs:
                varargs[sname] = kwargs[sname]
            else:
                if (sname=='edgecolors' and not i==1) or (i==1 and sname=='facecolors'):
                    varargs[sname] = 'none'
                else:
                    varargs[sname]=color  
    elif gradColor==1:
        # color by gradient
        varargs = {}

    if 'markersize' in kwargs:
        varargs['s']=kwargs['markersize']
#         if 'marker' in varargs:
#             # markers don't work with size, for some reason
#             varargs.pop('marker')
    return varargs
        
def plotSeries(df2:pd.DataFrame, gradColor:int, ax, cmapname:str, dx, dy, varargs:dict) :
    '''df2 is already sorted into x,y,c where c is color'''
    if len(df2)==0:
        return
    if gradColor==1:
        # gradient coloring
        sc = ax.scatter(df2['x'],df2['y'],linestyle='None',zorder=100,c=df2['c'],cmap=cmapname,**varargs)
        if 's' in varargs:
            varargs.pop('s')
        cmin = df2.c.min()
        cmax = df2.c.max()
        if dx>0 and dy>0:
            cmap = cm.get_cmap(cmapname)   
            for j, row in df2.iterrows():
                color = cmap((row['c']-cmin)/(cmax-cmin))
                sc = ax.errorbar([row['x']],[row['y']], xerr=[row['xerr']], yerr=[row['yerr']],linestyle='None', color=color,**varargs)
    else:
        sc = ax.scatter(df2['x'],df2['y'], **varargs)
        for s in ['label', 'facecolors', 'edgecolors', 's', 'fillstyle','marker']:
            if s in varargs:
                varargs.pop(s)
        sc = ax.errorbar(df2['x'],df2['y'], xerr=df2['xerr'], yerr=df2['yerr'],linestyle='None',marker='', elinewidth=1, **varargs)
    return sc

def idealLines(**kwargs) -> None:
    '''add vertical and horizontal lines'''
    if 'xideal' in kwargs:
        kwargs['ax'].axvline(kwargs['xideal'], 0,1, color='gray', linestyle='--', linewidth=1, label='ideal')
    if 'yideal' in kwargs:
        varargs = {'color':'gray', 'linestyle':'--', 'linewidth':'1'}
        if not 'xideal' in kwargs:
            varargs['label']='ideal'
        kwargs['ax'].axhline(kwargs['yideal'], 0,1, **varargs)
        
def scatterSS(ss:pd.DataFrame, xvar:str, yvar:str, colorBy:str, logx:bool=False, logy:bool=False, gradColor:int=0, dx:float=0.1, dy:float=1, cmapname:str='coolwarm', fontsize=10, plotReg:bool=False, grid:bool=True, **kwargs):
    '''scatter plot. 
    colorBy is the variable to color by. gradColor 0 means color by discrete values of colorBy, gradColor 1 means means to use a gradient color scheme by values of colorBy, gradColor 2 means all one color, one type of marker. gradColor 0 with 'color' in kwargs means make it all one color, but change markers.
    xvar is the x variable name, yvar is the y variable name. 
    logx=True to plot x axis on log scale. logy=True to plot y on log scale.
    dx>0 to group points by x and take error. otherwise, plot everything. dx=1 to average all points together'''

    plt.rc('font', size=fontsize) 
    
    if not (xvar in ss and yvar in ss):
        raise NameError(f'Variable name {xvar} or {yvar} is not in table')
    fig,ax,kwargs = setUpAxes(xvar, yvar, **kwargs)  # establish figure and axis
    setLog(ax, logx, logy)                    # set axes to log or not
               # get a colormap function
    ss1 = ss.copy()
    
    # remove NA from table
    if not colorBy in ss:
        logging.warning(f'variable {colorBy} is not in table')
        gradColor = 2
        ss1 = ss1[[xvar,yvar]].dropna()  
    else:
        ss1 = ss1[[xvar, yvar, colorBy]].dropna()
        
    # get a list of values on which to create series. if gradient color scheme, all one series
    if gradColor>0:
        lst = [0]
    else:
        lst = ss1[colorBy].unique()
        
    # iterate through list of series
    for i,val in enumerate(lst):
        varargs = seriesColor(gradColor, cmapname, i, lst, **kwargs)
        zvar = colorBy
        if colorBy in ss:
            zvar = colorBy
        if gradColor==0:
            ss2 = ss1[ss1[colorBy]==val]
            varargs['label']=val
        else:
            ss2 = ss1

        if len(ss2)>0:
            if grid:
                df2 = toGrid(ss2, xvar, yvar, zvar, logx, logy, dx, dy)
            else:
                df2 = toGroups(ss2, xvar, yvar, zvar, logx, logy, dx, dy)
            sc = plotSeries(df2, gradColor, ax, cmapname, dx, dy, varargs)
      
    # add verticals, horizontals
    idealLines(**kwargs)
        
    # add legends
    if gradColor==1 and colorBy in ss and not ('legend' in kwargs and kwargs['legend']==False):
        cbar = plt.colorbar(sc, label=colorBy)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels)>0 and not ('legend' in kwargs and kwargs['legend']==False):
            if len(fig.axes)==1 or ('legendloc' in kwargs and kwargs['legendloc']=='right'):
                ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', title=colorBy.replace('_',  ' '), frameon=False)
            else:
                ax.legend(bbox_to_anchor=(0,1), loc='lower left', title=colorBy.replace('_',  ' '), frameon=False)
                
    # set square
    setSquare(ax)
    fixTicks(ax, logx, logy)
    if plotReg:
        if logx:
            xvar = xvar+'_log'
        if logy:
            yvar = yvar+'_log'
        regressionSS(ss, xvar, yvar, ax)
    return fig,ax

def setSquare(ax):
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
def regressionSS(ss:pd.DataFrame, xvar:str, yvar:str, ax) -> None:
    '''add a linear regression to the plot'''
    if (not xvar in ss and xvar[-4:]=='_log'):
        ss = me.addLogs(ss, [xvar[:-4]])
    if (not yvar in ss and yvar[-4:]=='_log'):
        ss = me.addLogs(ss, [yvar[:-4]])
    ss2 = ss.copy()
    ss2.replace([np.inf, -np.inf], np.nan, inplace=True)  # remove infinite values
    ss2 = ss2.dropna(subset=[xvar,yvar])
    reg = rg.regPD(ss2, [xvar], yvar)
    min1 = ss2[xvar].min()
    max1 = ss2[xvar].max()
    logxlist = list(np.arange(min1, max1, (max1-min1)/20))
    ylist = [reg['c']+reg['b']*x for x in logxlist]
    if xvar[-3:]=='log':
        xlist = [10**i for i in logxlist]
    else:
        xlist = logxlist
    if yvar[-3:]=='log':
        ylist = [10**i for i in ylist]
    ylim = ax.get_ylim()
    xlist = [xlist[i] for i, y in enumerate(ylist) if ((y>ylim[0])&(y<ylim[1]))]
    ylist = [ylist[i] for i, y in enumerate(ylist) if ((y>ylim[0])&(y<ylim[1]))]
    ax.plot(xlist, ylist, color='black')
    ax.text(0.9, 0.9, 'r$^2$ = {:0.2}'.format(reg['r2']), transform=ax.transAxes)
    
def subFigureLabel(ax, label:str) -> None:
    '''add a subfigure label to the top left corner'''
    ax.text(0.05, 0.95, label, fontsize=12, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
    
def subFigureLabels(axs) -> None:
    '''add subfigure labels to all axes'''
    alphabet_string = string.ascii_uppercase
    alphabet_list = list(alphabet_string)
    if len(axs.shape)==1:
        # single row
        for ax in axs:
            subFigureLabel(ax, alphabet_list.pop(0))
    else:
        # 2d array
        for axrow in axs:
            for ax in axrow:
                subFigureLabel(ax, alphabet_list.pop(0))
    
    
def calcTicks(lim:Tuple[float]):
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
    
def fixTicks(ax, logx:bool, logy:bool):
    '''fix log scale ticks'''
    
    if logx:
        lim = ax.get_xlim()
        ticks, locmin = calcTicks(lim)
        ax.set_xticks(ticks)
        if min(ticks)>0.1 and max(ticks)<=10:
            ax.set_xticklabels(ticks)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if logy:
        lim = ax.get_ylim()
        ticks, locmin = calcTicks(lim)
        ax.set_yticks(ticks)
        if min(ticks)>0.1 and max(ticks)<=10:
            ax.set_yticklabels(ticks)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

def sweepTypeSS(ss:pd.DataFrame, xvar:str, yvar:str, cmapname:str='coolwarm', **kwargs):
    '''plot values based on sweep type'''
    
    ss.sort_values(by='sweepType')
    cmap = cm.get_cmap(cmapname)
    fig,ax,kwargs0 = setUpAxes(xvar, yvar, **kwargs)  # establish figure and axis
    if len(ss.sigma.unique())==1:
        # all the same surface tension: make visc blue and speed red
        for i,s in enumerate(['visc', 'speed']):
            ss0 = ss[ss.sweepType.str.startswith(s)]
            color0 = cmap(0.99*i)
            u = ss0.sweepType.unique()
            for j,st in enumerate(u):
                color = cmap(1*i + (0.4-i)*j/len(u))
                ma = getMarker(j, color)
                kwargs1 = {**kwargs0, **ma}
                scatterSS(ss0[ss0.sweepType==st], xvar, yvar, 'sweepType', cmapname=cmapname, **kwargs1)
                if 'yideal' in kwargs0:
                    kwargs0.pop('yideal')
                if 'xideal' in kwargs0:
                    kwargs0.pop('xideal')
    else:
        scatterSS(ss[(ss.sweepType.str.startswith('speed'))], xvar, yvar, 'sweepType', color='#555555', **kwargs0)
        if 'yideal' in kwargs0:
            kwargs0.pop('yideal')
        if 'xideal' in kwargs0:
            kwargs0.pop('xideal')
        scatterSS(ss[(ss.vRatio==1)], xvar, yvar, 'ink_type', **kwargs0)
    return fig, ax


def contourSS(ss:pd.DataFrame, xvar:str, yvar:str, zvar:str, logx:bool=False, logy:bool=False):
    '''contour plot with interpolation'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ss1 = ss.copy()
    X_unique = np.sort(ss1[xvar].unique())
    Y_unique = np.sort(ss1[yvar].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    Z = ss1.pivot_table(index=xvar, columns=yvar, values=zvar).T.values
    zmin = ss1[zvar].min()
    zmax = ss1[zvar].max()
    zmin = round(zmin, -int(np.log10(zmin))+1)
    zmax = round(zmax, -int(np.log10(zmax))+1)
    dz = round((zmax-zmin)/10, -int(np.log10(zmin))+1)
    levels = np.array(np.arange(zmin, zmax, dz))
    cpf = ax.contourf(X,Y,Z, len(levels), cmap=cm.coolwarm)
    line_colors = ['black' for l in cpf.levels]
    cp = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
    ax.clabel(cp, fontsize=10, colors=line_colors)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    cbar = plt.colorbar(cpf, label=zvar)
    return fig, ax


def getMeshTicks(piv:pd.pivot_table, x:bool, log:bool) -> Tuple[list,list]:
    '''get clean list of ticks and tick positions from pivot table'''
    if x:
        ylist = piv.columns
        n = piv.shape[1]
    else:
        ylist = piv.index
        n = piv.shape[0]
    if log:
        dy = np.log10(ylist[1])-np.log10(ylist[0])
        y0 = np.log10(ylist[0])
    else:
        dy = ylist[1]-ylist[0]
        y0 = ylist[0] # original y0, where ticks = m(x-0.5)+y0
    m = roundToOrder(dy) # clean m
    if log:
        logy0f = roundToOrder(round(np.log10(ylist[0])/m)*m) # clean y0
        labelsfi = [str(roundToOrder(logy0f+i*m)) for i in range(n)]
        labelsf = ["$10^{{{}}}$".format(i) for i in labelsfi]
        posf = [logy0f+i*m for i in range(n)]
        ticksf = [(y-y0)/m+0.5 for y in posf]
    else:
        y0f = round(ylist[0], int(ylist[0]/m))
        labelsf = [roundToOrder(y0f+i*m) for i in range(n)]
        ticksf = [(y-y0)/m+0.5 for y in labelsf]
    return ticksf, labelsf


def colorMeshSS(ss:pd.DataFrame, xvar:str, yvar:str, zvar:str, logx:bool=False, logy:bool=False, dx:float=0.1, dy:float=0.1, cmapname:str='coolwarm', **kwargs):
    '''contour plot with no interpolation'''
    
    if not (xvar in ss and yvar in ss):
        raise NameError('Variable name is not in table')
    if 'fig' in kwargs:
        fig = kwargs['fig']
    else:
        fig = plt.figure()
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(111)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ss2 = ss.copy()
    if len(ss2)==0:
        return fig, ax
    df2 = toGrid(ss2, xvar, yvar, zvar, logx, logy, dx, dy, rigid=(dx>0 or dy>0))
    piv = pd.pivot_table(df2, index='y', columns='x', values='c')
    sc = ax.pcolormesh(piv, cmap=cmapname)
    if dx>0:
        xpos, xticks = getMeshTicks(piv, True, logx)
    else:
        xpos = [0.5+i for i in range(len(piv.columns))]
        xticks = piv.columns
    ax.set_xticks(xpos, minor=False)
    ax.set_xticklabels(xticks, minor=False) 
    if dy>0:
        ypos, yticks = getMeshTicks(piv, False, logy)
    else:
        ypos = [0.5+i for i in range(len(piv.index))]
        yticks = piv.index
    
    ax.set_yticks(ypos, minor=False)
    ax.set_yticklabels(yticks, minor=False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            c = piv.iloc[i].iloc[j]
            if not pd.isna(c):
                ax.text(j+0.5,i+0.5,'{:0.2f}'.format(c), horizontalalignment='center',verticalalignment='center')
    if not ('legend' in kwargs and not kwargs['legend']): 
        cbar = plt.colorbar(sc, label=zvar)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_title(zvar)
    return fig,ax

def regRow(ssi:pd.DataFrame, xcol:str, ycol:str) -> dict:
    '''get regression and correlation info for a single x,y variable combo'''
    reg = rg.regPD(ssi, [xcol], ycol)
    spear = rg.spearman(ssi, xcol, ycol)
    reg['coeff'] = reg.pop('b')
    reg = {**reg, **spear}
    return reg

def regressionTable(ss:pd.DataFrame, yvar:str, logy:bool=True, printOut:bool=True, export:bool=False, exportFolder:str=os.path.join(cfg.path.fig, 'regressions'), **kwargs) -> List[pd.DataFrame]:
    ss0 = ss.copy()
    ss0.dropna(subset=[yvar], inplace=True)
    ss0 = ss0[ss0.ink_days==1]
    ss0 = ss0.sort_values(by='sigma')
    ssca1 = ss0.copy()
    ssca1 = ssca1[ssca1.sigma>0]
    sslap = ss0.copy()
    sslap = sslap[sslap.ink_base=='water']
    dflist = []
    for k, ssi in enumerate([ssca1, sslap]):
        dfall = []
        if len(ssi[yvar].unique())<2:
            if printOut:
                if k==0:
                    logging.info(f'All {yvar} values the same for nonzero surface tension\n---------------------------\n\n')
                else:
                    logging.info(f'All {yvar} values the same for zero surface tension\n---------------------------\n\n')
        else:
            # define y variables
            if logy:
                ssi = me.addLogs(ssi, [yvar])
                ycol = yvar+'_log'
            else:
                ycol = yvar

            # define x variables
            if k==0:
                varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Re', 'Bm', 'visc0']
            else:
                varlist = ['Re', 'Bm', 'visc0']

            # add logs and ratios
            for i,s1 in enumerate(['sup', 'ink']):
                ssi = me.addLogs(ssi, [s1+'_'+v for v in varlist])
            for i,s1 in enumerate(['Prod', 'Ratio']):
                ssi = me.addRatios(ssi, varlist=varlist, operator=s1)
                ssi = me.addLogs(ssi, [v+s1 for v in varlist])


            # go through each variable and get sup, ink, product, ratio
            for j, s2 in enumerate(varlist):
                df = []
                if s2=='dPR':
                    s2i = 'd_{PR'
                elif s2=='dnorm':
                    s2i = 'd_{Est}/d_{PR'
                elif s2=='visc0':
                    s2i = '\eta'
                else:
                    s2i = s2

                if s2=='Ca':
                    ssi = me.addLogs(ssi, ['int_Ca'])
                    reg = regRow(ssi, 'int_Ca_log', ycol)
                    reg['title'] = '$Ca$'
                    df.append(reg)

                # 2 variable correlation
    #             reg = rg.regPD(ssi, [f'ink_{s2}_log', f'sup_{s2}_log'], yvar)
    #             if s2i[-4:]=='_{PR':
    #                 reg['title'] = '$'+s2i+',ink}, '+s2i+',sup}$'
    #             else:
    #                 reg['title'] = '$'+s2i+'_{ink}, '+s2i+'_{sup}$'
    #             reg['coeff'] = ('{:0.2f}'.format(reg.pop('b0')),  '{:0.2f}'.format(reg.pop('b1')))
    #             df.append(reg)

                # single variable correlation
                for s1 in ['ink_', 'sup_']:
                    xcol = f'{s1}{s2}_log'
                    reg = regRow(ssi, xcol, ycol)
                    if s2i[-4:]=='_{PR':
                        reg['title']='$'+s2i+','+s1[:-1]+'}$'
                    else:
                        reg['title'] = '$'+s2i+'_{'+s1[:-1]+'}$'
                    df.append(reg)

                # products and ratios
                for s1 in ['Prod', 'Ratio']:
                    xcol = f'{s2}{s1}_log'
                    reg = regRow(ssi, xcol, ycol)
                    op = '' if s1=='Prod' else '/'
                    if s2i[-4:]=='_{PR':
                        s2ii = s2i+','
                    else:
                        s2ii = s2i+'_{'
                    reg['title'] = '$'+s2ii+'ink}'+op+s2ii+'sup}$'
                    df.append(reg)
                df = pd.DataFrame(df)

                # label best fit
                crit = ((abs(df.spearman_corr)>0.9*abs(df.spearman_corr).max())&(df.spearman_p<0.05)&(abs(df.spearman_corr)>0.5))
                df.spearman_p = ['{:0.1e}'.format(i) for i in df.spearman_p]
                df.spearman_corr = ['{:0.2f}'.format(i) for i in df.spearman_corr]
                df.r2 = ['{:0.2f}'.format(i) for i in df.r2]
                for sname in ['title', 'r2', 'spearman_corr', 'spearman_p']:
                    # bold rows that are best fit
                    df.loc[crit,sname] = ['$\\bm{'+(i[1:-1] if i[0]=='$' else i)+'}$' for i in df.loc[crit,sname]]
                if len(dfall)==0:
                    dfall = df
                else:
                    dfall = pd.concat([dfall, df])

            # combine into table
            df = dfall
            df = df[['title', 'r2', 'coeff', 'c', 'spearman_corr', 'spearman_p']]
            df = df.rename(columns={'r2': '$r^2$', 'title':'variables', 'coeff':'b', 'spearman_corr':'spearman coeff', 'spearman_p':'spearman p'})
            dflist.append(df)
            if 'nickname' in kwargs:
                nickname = kwargs['nickname']
            else:
                nickname = yvar
            shortcaption = f'Linear regressions for {nickname}'
            if k==0:
                shortcaption+=' at nonzero surface tension'
                label = f'tab:{yvar}RegNonZero'
            else:
                shortcaption+=' at zero surface tension'
                label = f'tab:{yvar}RegZero'
            longcaption = r'Table of linear regressions of log-scaled variables and Spearman rank correlations for \textbf{'+nickname+r'} at non-zero surface tension. For example, ${Re}_{ink}$ indicates a regression fit to $h/w = 10^c*Re_{ink}^b$. A Spearman rank correlation coefficient of -1 or 1 indicates a strong correlation. Variables are defined in table \ref{tab:variableDefs}.'

            dftext = df.to_latex(index=False, escape=False, float_format = lambda x: '{:0.2f}'.format(x) if pd.notna(x) else '' , caption=(longcaption, shortcaption), label=label)
            dftext = dftext.replace('\\toprule\n', '')
            dftext = dftext.replace('\\midrule\n', '')
            dftext = dftext.replace('\\bottomrule\n', '')
            dftext = dftext.replace('\begin{table}', '\begin{table}[H]')
            ctr = -10
            dftextOut = ''
            for line in iter(dftext.splitlines()):
                dftextOut = dftextOut+line+'\n'
                ctr+=1
                if 'variables' in line:
                    ctr = 0
                if 'bm{Ca}' in line or '$Ca$' in line:
                    ctr = 0
                if ctr==4 and not line.startswith('\\end'):
                    dftextOut = dftextOut+'\t\t\\hline\n'
                    ctr=0
            if printOut:
                print(dftextOut)
            if export:
                fn = os.path.join(exportFolder, label[4:]+'.tex')
                file2 = open(fn ,"w")
                file2.write(dftextOut)
                file2.close()
                logging.info(f'Exported {fn}\n---------------------------\n\n')
    return dflist