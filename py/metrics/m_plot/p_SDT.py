'''Functions for plotting still and video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.cm as cm
import matplotlib.colors as mc
import colorsys
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import py.metrics.m_SDT as me
import py.metrics.m_plot.m_plots as mp
from tools.config import cfg

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

def shrinkagePlot(ms, fstri:str, export:bool=True) -> mp.multiPlot:
    '''plot the lengthening over time, change in length, asymmetry, and change during disturbance for one folder
    ms is a metricSummary object
    fstri is the folder to take the single time series from
    fn is the export file name
    '''
    if 'disturbHoriz' in fstri:
        orie = 'HOP'
        var = 'wn'
        folderClass = me.folderHorizSDT
    elif 'disturbVert' in fstri:
        orie = 'V'
        var = 'hn'
        folderClass = me.folderVertSDT
    elif 'disturbUnder' in fstri:
        orie = 'HIP'
        var = 'wn'
        folderClass = me.folderUnderSDT

    yvl = mp.multiSpecific(ms, ms.ss, xvars=[['sup_Oh' for i in range(3)], ['sup_Oh' for i in range(3)]]
                       , yvars=[[f'd{var}dt_w1o', f'd{var}dt_d1o', f'd{var}dt_w2o'], [f'delta_{var}_disturb1', 'ldiff_w2o', '']]
                       , cvar='spacing', plotType='paper', yideal=me.ideals(), sharey=False, sharex=False, legendAbove=True, tightLayout=False
                   , logx=True, logy=False, mode='scatter', dx=0.15, holdPlots=True, xlim={'sup_Oh':[9, 300], 'sup_dnorma':[0.3, 30]})
    [yvl.shareAxes(0,i,0,0,s) for s in ['x','y'] for i in [1,2]]
    [yvl.shareAxes(1,i,1,0,s) for s in ['x','y'] for i in [1]]


    yvl.plots()   # plot the data
    for i in [0,1]:
        yvl.axs[1,i].set_xticks([1, 10])
        yvl.axs[1,i].yaxis.set_minor_locator(MultipleLocator(0.05))
        yvl.axs[1,i].xaxis.set_major_formatter('{x:.0f}')
    for i,j in [(0,0), (0,1), (0,2), (1,0), (1,1)]:
        yvl.axs[i,j].set_xticks([10, 100])
        yvl.axs[i,j].set_xlim([9, 400])
        yvl.axs[i,j].yaxis.set_minor_locator(MultipleLocator(0.05))
        yvl.axs[i,j].xaxis.set_major_formatter('{x:.0f}')
    # yvl.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    folder = os.path.join(cfg.path.server, fstri)
    fv = folderClass(folder, overwriteMeasure=False, overwriteSummary=False, diag=0, overrideSegment=False)  # import the summary
    fv.summarize();
    fv.plotValue(var, xvar='wtime', ax=yvl.axs[1,2], fontsize=8, legend=True, legendLoc='annotate')  # plot the time series
    yvl.axs[1,2].set_xlim([0, fv.df.wtime.max()+1])
    #yvl.export(os.path.join(cfg.path.fig, 'SDT', 'plots', 'shrinkage_HOP'))
    for i in range(3):
        yvl.axs[0,i].set_ylabel('lengthening/time (intended $L$/s)', fontsize=8)
    yvl.axs[1,0].set_ylabel('$\Delta$length (intended $L$)', fontsize=8)
    yvl.axs[1,1].set_ylabel('Normalized asymmetry', fontsize=8)
    yvl.axs[1,2].set_ylabel('Length (intended $L$)', fontsize=8)
    yvl.axs[0,0].set_title('Relax after writing line 1', fontsize=8)
    yvl.axs[0,1].set_title('Relax after disturbing line 1', fontsize=8)
    yvl.axs[0,2].set_title('Relax after writing line 2', fontsize=8)
    yvl.axs[1,0].set_title('While disturbing line 1', fontsize=8)
    yvl.axs[1,1].set_title('After writing line 2', fontsize=8)
    if export:
        yvl.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'shrinkage_{orie}'))
    return yvl

def shiftPlot(ms, orie:str, xvar:str='tau0aRatio', export:bool=False) -> mp.multiPlot:
    '''plot the shift in position'''
    if orie=='HOP':
        near = 'yTop'
        far = 'yBot'
        nearstr = 'y_{top}'
        farstr= 'y_{bot}'
    elif orie=='HIP':
        near = 'y0'
        far = 'yf'
        nearstr = 'y_{near}'
        farstr = 'y_{far}'
    elif orie=='V':
        near = 'xf'
        far = 'x0'
        nearstr = 'x_{near}'
        farstr = 'x_{far}'
    yvl = mp.multiSpecific(ms, ms.ss, xvars=[[xvar for i in range(3)] for j in range(2)]
                       , yvars=[[f'{var}_w1o', f'delta_{var}_disturb1', f'{var}_w2o'] for var in [near, far]]
                       , cvar='spacing', plotType='paper', yideal=me.ideals(), sharey=False, sharex=True, legendAbove=True, tightLayout=False
                   , logx=True, logy=False, mode='scatter', dx=0.15, holdPlots=True)
    [yvl.shareAxes(i,j,0,0,s) for s in ['x','y'] for (i,j) in [(0,2), (1,0), (1,2)]]
    [yvl.shareAxes(0,1,1,1,s) for s in ['x', 'y']]
    yvl.plots()
    [yvl.axs[0,j].set_ylabel('$'+nearstr+'$ ($d_{est}$)', fontsize=8) for j in [0,2]]
    [yvl.axs[1,j].set_ylabel('$'+farstr+'$ ($d_{est}$)', fontsize=8) for j in [0,2]]
    [yvl.axs[0,j].set_ylabel('$\Delta '+nearstr+'$ ($d_{est}$)', fontsize=8) for j in [1]]
    [yvl.axs[1,j].set_ylabel('$\Delta '+farstr+'$ ($d_{est}$)', fontsize=8) for j in [1]]
    [yvl.axs[i,0].set_title('After writing line 1', fontsize=8) for i in [0,1]]
    [yvl.axs[i,1].set_title('While disturbing line 1', fontsize=8) for i in [0,1]]
    [yvl.axs[i,2].set_title('After writing line 2', fontsize=8) for i in [0,1]]
    
    for j in [0,1,2]:
        for i in [0,1]:
            if xvar=='tau0aRatio':
                yvl.axs[i,j].set_xticks([0.03, 0.1, 0.3])
                yvl.axs[i,j].xaxis.set_major_formatter('{x:.2f}')
            elif xvar=='ink_tau0a':
                yvl.axs[i,j].set_xticks([1, 5, 10])
                yvl.axs[i,j].xaxis.set_major_formatter('{x:.0f}')
            yvl.axs[i,j].yaxis.set_minor_locator(MultipleLocator(0.1))
    if export:
        yvl.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'shift_{orie}'))
    return yvl

def shiftPlotXS(ms, orie:str, xvar:str='tau0aRatio', export:bool=False) -> mp.multiPlot:
    '''plot the shift in position'''
    near = 'yTop'
    far = 'yBot'
    yvl = mp.multiSpecific(ms, ms.ss, xvars=[[xvar] for i in range(4)]
                       , yvars=[[f'delta_{var}_{change}'] for var in [near, far] for change in ['disturb1', 'write2']]
                       , cvar='spacing', plotType='paper', yideal=me.ideals(), sharey=True, sharex=True, legendAbove=False, tightLayout=True
                   ,logx=True, logy=False, mode='scatter', dx=0.15, holdPlots=False, figsize=(2.5, 8))
    for j in [0,1,2,3]:
        yvl.axs[j,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    if export:
        yvl.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'shift_xs_{orie}'))
    return yvl

def fusionPlot(ms, orie:str, export:bool=False) -> mp.scatterPlot:
    '''plot fusion between filaments'''
    sp = mp.scatterPlot(ms, ms.ss, xvar='tau0aRatio', yvar='roughness_w2o', cvar='spacing', logx=True, plotType='paper', figsize=(3*6.5/7.2, 2.5*6.5/7.2))
    # sp.ax.set_xticks([30, 100, 300])
    # sp.ax.set_xticklabels([30, 100, 300])
    sp.ax.set_ylim([0, 1])
    sp.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    sp.ax.set_ylabel('Roughness after writing line 2', fontsize=8)
    if orie=='HIP':
        sp.ax.set_title('Horizontal in plane', fontsize=8)
    elif orie=='HOP':
        sp.ax.set_title('Horizontal out of plane', fontsize=8)
    elif orie=='V':
        sp.ax.set_title('Vertical', fontsize=8)
    if export:
        sp.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'fusion_{orie}'))
    return sp

def fusion3Plot(ms, orie:str, export:bool=False) -> mp.scatterPlot:
    '''plot fusion between filaments'''
    yvl = mp.multiSpecific(ms, ms.ss, xvars=[['sup_Oh'] for i in range(2)]
                       , yvars=[[f'delta_roughness_{change}'] for change in ['disturb2', 'write3']]
                       , cvar='spacing', plotType='paper', yideal=me.ideals(), sharey=False, sharex=True
                           , legendAbove=False, tightLayout=True
                   ,logx=True, logy=False, mode='scatter', dx=0.15, holdPlots=False, figsize=(2.25, 4))
    for i,axrow in enumerate(yvl.axs):
        for ax in axrow:
            ax.set_xticks([30, 100, 300])
            ax.set_xticklabels([30, 100, 300])
            if i==0:           
                ax.set_ylim([-0.35, 0.1])
                ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0])
            else:
                ax.set_ylim([-0.05, 0.9])
                ax.set_yticks([0, 0.2, 0.4,0.6, 0.8])
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    if orie=='HIP':
        yvl.axs[0,0].set_title('Horizontal in plane', fontsize=8)
    elif orie=='HOP':
        yvl.axs[0,0].set_title('Horizontal out of plane', fontsize=8)
    elif orie=='V':
        yvl.axs[0,0].set_title('Vertical', fontsize=8)
    if export:
        yvl.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'fusion3_{orie}'))
    return yvl

def gapPlot(ms, orie:str, xvar:str='spacing', yvar:str='space_b_d1p', export:bool=False, **kwargs) -> mp.scatterPlot:
    '''plot the gap between the nozzle and filament'''
    sp = mp.scatterPlot(ms, ms.ss, xvar=xvar, yvar=yvar, cvar='vRatio', logx=False, plotType='paper', figsize=(2.5, 2.5), **kwargs)
    sp.ax.set_xlabel('spacing ($d_{est}$)')
    sp.ax.set_ylabel('gap ($d_{est}$)')
    sp.ax.set_ylim([-0.02, 0.53])
    sp.ax.set_xlim([0.45, 1.3])
    sp.ax.set_yticks(np.arange(0, 0.75, 0.25))
    sp.ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    sp.ax.set_xticks([0.5, 0.75, 1, 1.25])
    sp.ax.xaxis.set_minor_locator(MultipleLocator(0.125))
    t1 = {'HIP':'Horizontal in plane', 'HOP':'Horizontal out of plane', 'V':'Vertical'}[orie]
    t2 = {'space_b_d1p':'Disturbing line 1', 'space_b_w2p':'Writing line 2'}[yvar]
    title = f'{t1}\n{t2}'
    sp.ax.set_title(title, fontsize=8)
    xr = np.arange(0.5, 2.5, 0.01)
    if orie=='HOP':
        vlist = [1]
    else:
        vlist = [0.56, 1, 2.25]
    for vratio in vlist:
        color = {2.25:'#b40426', 1:'#808080', 0.56:'#3b4cc0'}[vratio]
        if orie=='HIP':
            dest = 0.603*np.sqrt(vratio)  # mm
            do = 0.907  # mm
            scrit = 0.5+do/(2*dest)
        else:
            scrit = 1
        yr = [0 if x<=(scrit) else x-(scrit) for x in xr]
        sp.ax.plot(xr, yr, color=color, linestyle='dashed', linewidth=1)
    if export:
        sp.export(os.path.join(cfg.path.fig, 'SDT', 'plots', f'gap_{orie}_{yvar}'))
    return sp
    