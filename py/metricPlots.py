#!/usr/bin/env python
'''Functions for plotting video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import seaborn as sns

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

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

def cubehelix1(val:float):
    '''val should be 0-1. returns a color'''
    cm = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
    return cm(val)


def adjust_lightness(color, amount=0.5):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def roundToOrder(val:float):
    if abs(val)<10**-14:
        return 0
    else:
        return round(val, -int(np.log10(abs(val)))+1)


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
    '''convert the data to an evenly spaced grid'''
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
                    c0, _ = pooledSE(ss3, zvar)
                else:
                    c0 = 0
                df2.append({'x':xm, 'xerr':xerr, 'y':ym, 'yerr':yerr, 'c':c0})
    df2 = pd.DataFrame(df2)
    return df2
   

def scatterSS(ss:pd.DataFrame, xvar:str, yvar:str, colorBy:str, logx:bool=False, logy:bool=False, gradColor:bool=False, dx:float=0.1, dy:float=1, cmapname:str='coolwarm', **kwargs):
    '''scatter plot. 
    colorBy is the variable to color by. gradColor means to use a gradient color scheme, otherwise use discrete colors.
    xvar is the x variable name, yvar is the y variable name. 
    logx=True to plot x axis on log scale. logy=True to plot y on log scale.
    dx>0 to group points by x and take error. otherwise, plot everything. dx=1 to average all points together'''
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
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    cmap = cm.get_cmap(cmapname)
    ss1 = ss.copy()
    if not colorBy in ss:
        gradColor = True
        ss1 = ss1[[xvar,yvar]].dropna()  # remove NA from table
    else:
        ss1 = ss1[[xvar, yvar, colorBy]].dropna()
    if gradColor:
        lst = [0]
    else:
        lst = ss1[colorBy].unique()
    for i,val in enumerate(lst):
        if not gradColor:
            varargs = {'fmt':'o'}
            ss2 = ss1[ss1[colorBy]==val]
            if len(lst)>1:
                color = cmap((i)/(len(lst)-1))
                if i/(len(lst)-1)==0.5 and cmapname=='coolwarm':
                    color = adjust_lightness(color, 0.5) # darken middle color
            else:
                color = 'black'
            varargs['color']=color   
            varargs['label']=val
        else: 
            varargs = {}
            ss2 = ss1
        if gradColor and colorBy in ss: 
            zvar = colorBy
            
        else:
            zvar = ''
        if len(ss2)>0:
            df2 = toGrid(ss2, xvar, yvar, zvar, logx, logy, dx, dy)
            if len(df2)>0:
                if gradColor and colorBy in ss:
                    cmin = df2.c.min()
                    cmax = df2.c.max()
                    if dx>0 and dy>0:
                        for j, row in df2.iterrows():
                            color = cmap((row['c']-cmin)/(cmax-cmin))
                            sc = ax.errorbar([row['x']],[row['y']], xerr=[row['xerr']], yerr=[row['yerr']],linestyle='None', color=color,**varargs)
                    sc = ax.scatter(df2['x'],df2['y'],linestyle='None',zorder=100,c=df2['c'],cmap=cmapname,**varargs)
                else:
                    sc = ax.errorbar(df2['x'],df2['y'], xerr=df2['xerr'], yerr=df2['yerr'],linestyle='None',**varargs)
                    if not colorBy in ss:
                        sc = ax.scatter(df2['x'],df2['y'],linestyle='None',zorder=100,**varargs)
        
    if 'xideal' in kwargs:
        ax.axvline(kwargs['xideal'], 0,1, color='gray', linestyle='--', linewidth=1, label='ideal')
    if 'yideal' in kwargs:
        ax.axhline(kwargs['yideal'], 0,1, color='gray', linestyle='--', linewidth=1, label='ideal')
    if gradColor and colorBy in ss:
        cbar = plt.colorbar(sc, label=colorBy)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels)>0 and not ('legend' in kwargs and kwargs['legend']==False):
            if len(fig.axes)==1:
                ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', title=colorBy)
            else:
                ax.legend(bbox_to_anchor=(0,1), loc='lower left', title=colorBy)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return fig,ax


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
    return fig,ax