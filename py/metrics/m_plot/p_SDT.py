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

def shrinkagePlot(ms, fstri:str, export:bool=True) -> None:
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
    yvl = mp.multiSpecific(ms, ms.ss, xvars=[['sup_Oh' for i in range(3)], ['sup_dnorma' for i in range(3)]]
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