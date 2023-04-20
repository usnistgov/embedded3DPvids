#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a whole folder'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
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
        super().__init__(folder, **kwargs)
        if not f'disturb' in os.path.basename(self.folder):
            return
        if 'pg' in kwargs:
            self.pg = kwargs['pg']
        else:
            self.pg = getProgDims(self.folder)
            self.pg.importProgDims()
        self.lines = list(self.pg.progDims.name)    
        
    def depvars(self) -> list:
        '''find the dependent variables measured by the function'''
        self.importMeasure()
        if len(self.df)>0:        
            dv = list(self.df.keys())
            dv.remove('line')
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
                out = out + [[f'w1o2', f'd1o1', 'disturb']]   # compare write observation 2 to disturb observation 1
            elif '_2_' in os.path.basename(self.folder):
                # 2 write, 1 disturb
                out = out + [[f'{ll}o{on}' for on in [1,2]]+[f'{ll}relax'] for ll in ['w1', 'w2', 'd2']]   # compare observation 1 and 2 for write and disturb
                out = out + [[f'w1o1', f'w2o1', 'write2']]   # compare write 1 observation 1 to write 2 observation 1
                out = out + [[f'w2o2', f'd2o1', 'disturb']]   # compare write 2 observation 2 to disturb observation 1
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
        out = [f'{ll}{on}' for ll in llist for on in ['p', 'o']]
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
    
    def plotValue(self, yvar:str, xvar:str='wtime') -> None:
        '''plot values over time'''
        if not yvar in self.df or not xvar in self.df:
            return
        fig, ax = plt.subplots(figsize=(8,6))
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
            dfa = df[df.pname.str.contains('p')]
            dfb = df[df.pname.str.contains('o')]
            for j,dfi in enumerate([dfa, dfb]):
                if j==0:
                    kwargs = {'label':label[0]}
                else:
                    kwargs = {}
                ax.scatter(dfi[xvar], dfi[yvar], marker=marker, color=color, facecolors=mfc, **kwargs)
                ax.plot(dfi[xvar], dfi[yvar], color=color, linestyle=linestyle)

        ux = self.du[xvar]
        if xvar=='time':
            ax.set_xlabel(f'Time since action ({ux})')
        elif xvar=='wtime':
            ax.set_xlabel(f'Time since writing ({ux})')
        else:
            ax.set_xlabel(f'{xvar} ({ux})')
        u = self.du[yvar]
        ax.set_ylabel(f'{yvar} ({u})', rotation=90)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.legend()
        fig.tight_layout()
