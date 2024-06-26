#!/usr/bin/env python
'''Functions for collecting data from stills of single double triple lines, for a whole folder'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import subprocess
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from folder_metric import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
   
class folderSDT(folderMetric):
    '''for a folder, measure the SDT lines
    export a table of values (Measure)
    export a list of failed files (Failures)
    export a row of summary values (Summary)
    '''
    
    def __init__(self, folder:str, **kwargs) -> None:
        super().__init__(folder, splitGroups=True, **kwargs)
        if not f'disturb' in os.path.basename(self.folder):
            return
        if not hasattr(self, 'pg'):
            if 'pg' in kwargs:
                self.pg = kwargs['pg']
            else:
                self.pg = getProgDimsPV(self.pv)
                self.pg.importProgDims()
        self.lines = list(self.pg.progDims.name)    
        
    def importMeasure(self):
        '''import the measure table'''
        super().importMeasure()
        self.df = self.df[~(self.df.pname=='o8')]  # remove 8th observation bc frame rate too slow, often blurry
        self.df = self.df[~(self.df.pname=='p5')]  # remove 5th printing step bc frame rate too slow, often blurry
        
    def depvars(self) -> list:
        '''find the dependent variables measured by the function'''
        self.importMeasure()
        if len(self.df)>0:        
            dv = list(self.df.keys())
            for s in ['line', 'gname', 'ltype', 'pr', 'pname', 'time', 'wtime', 'zdepth', 'usedML']:
                if s in dv:
                    dv.remove(s)
            return dv
        else:
            return []
    
    def pglines(self, name:str) -> pd.DataFrame:
        '''progdims names that contain this tag'''
        return self.pg.progDims[self.pg.progDims.name.str.contains(name)]
    
    def pgline(self, name:str) -> pd.Series:
        '''progDims line with name that matches this tag'''
        wodf = self.pglines(name)
        if not len(wodf)==1:
            return []
        else:
            return wodf.iloc[0]
    
    def pairTime(self, pair:list) -> float:
        '''get the time in seconds between the pair of images'''
        p1 = self.pgline(pair[0])
        p2 = self.pgline(pair[1])
        if len(p1)==0 or len(p2)==0:
            raise ValueError(f'Could not find pair {pair} in progDims')
        t1 = p1['tpic']
        t2 = p2['tpic']
        dt = t2-t1
        return dt
    
    def pairs(self) -> list:
        '''get a list of pairs to compare and a 3rd value that describes in words what we're evaluating'''
        out = []
        if 'xs' in os.path.basename(self.folder).lower():
            if '_1_' in os.path.basename(self.folder):
                # 1 write, 1 disturb
                out = out + [[f'{ll}o{on}' for on in [1,2]]+[f'{ll}relax'] for ll in ['w1', 'd1']]   # compare observation 1 and 2 for write and disturb
                out = out + [[f'w1o2', f'd1o1', 'disturb1']]   # compare write observation 2 to disturb observation 1
            elif '_2_' in os.path.basename(self.folder):
                # 2 write, 1 disturb
                out = out + [[f'{ll}o{on}' for on in [1,2]]+[f'{ll}relax'] for ll in ['w1', 'w2', 'd2']]   # compare observation 1 and 2 for write and disturb
                out = out + [[f'w1o1', f'w2o1', 'write2']]   # compare write 1 observation 1 to write 2 observation 1
                out = out + [[f'w2o2', f'd2o1', 'disturb2']]   # compare write 2 observation 2 to disturb observation 1
            elif '_3_' in os.path.basename(self.folder):
                # 3 write
                out = out + [[f'{ll}o{on}' for on in [1,2]]+[f'{ll}relax'] for ll in ['w1', 'w2', 'w3']]   # compare observation 1 and 2 for write and disturb
                out = out + [[f'w1o1', f'w2o1', 'write2']]   # compare write 1 observation 1 to write 2 observation 1
                out = out + [[f'w2o1', f'w3o1', 'write3']]   # compare write 2 observation 1 to write 3 observation 1
            else:
                raise ValueError(f'Unexpected shopbot file name in {self.folder}')
        else:
            wg = self.writeGroups()
            out = out + [[f'{ll}{on}' for on in ['p', 'o']]+[f'{ll}relax'] for ll in wg]   # compare in-progress and observed stats
            
            # compare values between observations
            for t in list(zip(['a']+wg, wg))[1:]:
                t0 = t[0]
                t1 = t[1]
                if t1[0]=='w':
                    # write move
                    out = out + [[f'{t0}o', f'{t1}o', f'write{t[1][1]}']]
                else:
                    out = out + [[f'{t0}o', f'{t1}o', f'disturb{t[1][1]}']]
        return out
    
    def writeGroups(self):
        '''names of groups of lines'''
        if '_1_' in self.folder:
            # 1 write, 1 disturb
            llist = ['w1', 'd1']   # measure written and disturbed during and just after writing
        elif '_2_' in self.folder:
            # 2 write, 1 disturb
            llist = ['w1', 'w2', 'd2']   # measure written and disturbed during and just after writing
        elif '_3_' in self.folder:
            # 3 write
            llist = ['w1', 'w2', 'w3']
        else:
            raise ValueError(f'Unexpected shopbot file name in {self.folder}')
        return llist
        

    def singles(self) -> list:
        '''get a list of single values to average across all 4 groups'''
        llist = self.writeGroups()
        if 'xs' in os.path.basename(self.folder).lower():
            slist = ['', 'o']
        else:
            slist = ['p', 'o']
        out = [f'{ll}{on}' for ll in llist for on in slist]
        return out
    
    def slopes(self) -> list:
        '''get a list of groups to measure slopes on'''
        llist = self.writeGroups()
        out = [f'{ll}o' for ll in llist]
        return out
    
    def ooChanges(self) -> list:
        '''list of observe-observe changes to collect'''
        llist = self.writeGroups()
        llist = [f'{ll}o' for ll in llist]
        return list(zip(llist, llist[1:]))
    
    def poChanges(self) -> list:
        '''list of progress-observe changes to collect'''
        llist = self.writeGroups()
        out = [(f'{ll}p', f'{ll}o') for ll in llist]
    
    def plotValue(self, yvar:str, xvar:str='wtime', fn:str='', figsize:tuple=(8,6), fontsize:int=8, legend:bool=True, legendLoc:str='inset', **kwargs) -> None:
        '''plot values over time'''
        plt.rcParams.update({'font.size': 8})
        if not hasattr(self, 'df'):
            self.importMeasure()
        if not yvar in self.df or not xvar in self.df:
            return
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else:
            fig, ax = plt.subplots(figsize=figsize)
        df2 = self.df.sort_values(by='wtime')
        colors = ['#033F63', '#c7a63a', '#7C9885', '#e391c1']
        
        weights = {'w1':'-', 'w2':'--', 'w3':'-.', 'd1':':', 'd2':':', 'd3':':'}
        markers = {'1':'o', '2':'s', '3':'*'}
        for label, df in df2.groupby(['pr', 'ltype']):
            marker = markers[label[1][1]]
            linestyle = weights[label[1]]
            color = colors[int(label[0][1])]
            if label[1][0]=='w':
                mfc = color
                edgecolor = 'none'
            else:
                mfc = 'none'
                edgecolor = color
            dfa = df[df.pname.str.contains('p')]   # printing
            dfb = df[df.pname.str.contains('o')]  # observing
            for j,dfi in enumerate([dfa, dfb]):
                if j==0:
                    kwargs = {'label':label[0]}
                else:
                    kwargs = {}
                xl = dfi[xvar]
                yl = dfi[yvar]
                if not pd.isna(yl.iloc[0]):
                    ax.scatter(xl, yl, marker=marker, color=color, facecolors=mfc, s=10, **kwargs)
                    ax.plot(xl, yl, color=color, linestyle=linestyle)
                    if legend and legendLoc=='annotate':
                        lnum = label[0][1]
                        if label[0][2]=='w':
                            ax.text(xl.iloc[-1]+1, yl.iloc[-1], f'depth {lnum}', fontsize=fontsize, color=color, ha='left')
                        else:
                            ax.text(xl.iloc[0]-1, yl.iloc[0], f'depth {lnum}', fontsize=fontsize, color=color, ha='right')

        ux = self.du[xvar]
        if xvar=='time':
            ax.set_xlabel(f'Time since action ({ux})', fontsize=fontsize)
        elif xvar=='wtime':
            ax.set_xlabel(f'Time since writing ({ux})', fontsize=fontsize)
        else:
            ax.set_xlabel(f'{xvar} ({ux})', fontsize=fontsize)
        u = self.du[yvar]
        ax.set_ylabel(f'{yvar} ({u})', rotation=90, fontsize=fontsize)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        if legend and not legendLoc=='annotate':
            ax.legend(fontsize=fontsize, frameon=False)
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.xaxis.set_major_formatter('{x:.0f}')

        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
      #  fig.tight_layout()
        if len(fn)>0 and os.path.exists(os.path.dirname(fn)):
            fn0 = os.path.splitext(fn)[0]
            for s in ['.png', '.svg']:
                fig.savefig(f'{fn0}{s}', bbox_inches='tight', dpi=300)
            logging.info(f'Exported {fn0}.png and .svg')
    