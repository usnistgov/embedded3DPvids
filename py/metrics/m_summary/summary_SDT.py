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
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from summary_metric import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class summarySDT(summaryMetric):
    '''holds data and functions for handling metric summary tables for disturbed lines'''
    
    def __init__(self, file:str, diag:bool=False):
        super().__init__(file)
        self.file = file
        if 'xs' in self.file.lower():
            self.type = 'xs'
        elif 'vert' in self.file.lower():
            self.type = 'vert'
        elif 'horiz' in self.file.lower():
            self.type = 'horiz'
        else:
            raise ValueError('Cannot identify print type')
        self.importStillsSummary(diag=diag)
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        '''import the stills summary and convert sweep types, capillary numbers'''
        self.ss,self.u = plainIm(self.file, ic=False)
        self.flipInv()
        for col in ['ink_surfactantWt', 'sup_surfactantWt']:
            self.ss[col] = self.ss[col].fillna(0)
        if diag:
            self.printIndeps()
            print()
            self.printKeyTable()
        return self.ss,self.u
    
    def strip(self, s:str) -> None:
        '''strip the decorations from the variable and get the initial variable that was measured by fileMeasure'''
        spl = re.split('_', s)
        for si in ['w1p', 'w1o', 'w2p', 'w2o','w3p', 'w3o'
                  , 'd1p', 'd1o', 'd2p', 'd2o', 'd3p', 'd3o'
                  , 'delta', 'disturb1', 'disturb2', 'disturb3'
                  , 'write1', 'write2', 'write3'
                  , 'w1relax', 'w2relax', 'w3relax'
                  , 'd1relax', 'd2relax', 'd3relax']:
            if si in spl:
                spl.remove(si)
        if len(spl)>1:
            if 'space' in spl:
                return '_'.join(spl)
            else:
                raise ValueError(f'Unexpected value {s} passed to summarySDT.strip')
        s2 = spl[0]
        if s2[-2:]=='dt':
            return s2[1:-2]
        else:
            return s2
        
    
    def printKeyTable(self): 
        '''print the dependent variables gridded by title'''
        c = []
        for var in self.depVars():
            s = self.strip(var)
            if not s in c:
                c.append(s)
        print('\033[1mDependents:\033[0m ', ', '.join(c))
        table = pd.DataFrame({'wp':dict([[i, f'X_w{i}p'] for i in range(3)]), 'wo':dict([[i, f'X_w{i}o'] for i in range(3)]), 'dw/dt':dict([[i, f'dXdt_w{i}o'] for i in range(3)]), 'wrelax':dict([[i, f'delta_X_w{i}relax'] for i in range(3)]), 'write':dict([[i, f'delta_X_write{i}'] for i in range(1,3)]), 'dp':dict([[i, f'X_d{i}p'] for i in range(2)]), 'do':dict([[i, f'X_d{i}o'] for i in range(2)]), 'dd/dt':dict([[i, f'dXdt_d{i}o'] for i in range(2)]), 'drelax':dict([[i, f'delta_X_d{i}relax'] for i in range(2)]), 'disturb':dict([[i, f'delta_X_disturb{i}'] for i in range(2)])})
        table.fillna('', inplace=True)
        display(table.T)
        

    def addRatios(self, ss:pd.DataFrame, **kwargs) -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        return super().addRatios(ss, self.firstDepCol(), **kwargs)

    def addLogs(self, ss:pd.DataFrame, varlist:List[str], **kwargs) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        return super().addLogs(ss, self.firstDepCol(), varlist, **kwargs)
    
    def depVarSpl(self, s:str) -> str:
        '''split the dependent variable to convert it to a new name'''
        spl = re.split('_', s)
        out = ''
        linename = spl[-1]
        try:
            linenum = int(linename[1])
        except ValueError:
            linenum = int(linename[-1])
        linenum = {1:'1st', 2:'2nd', 3:'3rd'}[linenum]
        if linename[0]=='w':
            if linename.endswith('relax'):
                out = out+f'{linenum} write relax'
            elif linename.startswith('write'):
                out = out+f'{linenum} write'
            elif linename[-1]=='p':
                out = out+f'{linenum} writing'
            elif linename[-1]=='o':
                out = out+f'{linenum} written'
        elif linename[0]=='d':
            if linename.endswith('relax'):
                out = out+f'{linenum} disturb relax'
            elif linename.startswith('disturb'):
                out = out+f'{linenum} disturb'
            elif linename[-1]=='p':
                out = out+f'{linenum} disturbing'
            elif linename[-1]=='o':
                out = out+f'{linenum} disturbed'
        out = out + ' '
        if spl[0]=='delta':
            out = out + '$\Delta$'
            var = spl[1]
        else:
            var = spl[0]
        if self.type=='vert':
            varlist = {'x0':'$x_{left}$', 'dxprint':'x shift under nozzle', 'segments':'segments', 'w':'width', 'h':'length', 'xf':'$x_{right}$', 'xc':'$x_{center}$', 'roughness':'roughness', 'emptiness':'emptiness', 'meanT':'ave thickness', 'stdevT':'stdev thickness', 'minmaxT':'thickness variation', 'ldiff':'left shrinkage', 'dx0dt':'left edge shift/time', 'dwdt':'widening/time', 'dhdt':'lengthening/time', 'dxfdt':'$\Delta x_{right}$/time', 'dxcdt':'$\Delta x_{center}$/time', 'dsegmentsdt':'rupturing/time', 'dldiffdt':'evening/time', 'droughnessdt':'roughening/time', 'demptinessdt':'emptying/time', 'dmeanTdt':'thickening/time', 'dstdevTdt':'d(stdev thickness)/dt', 'dminmaxTdt':'d(thickness variation)/time'}
            if var in varlist:
                out = out + varlist[var]
            else:
                out = out + var
        return out
 
            
    
    def varSymbol(self, s:str, lineType:bool=True, commas:bool=True, **kwargs) -> str:
        '''get a symbolic representation of the variable
        lineType=True to include the name of the line type in the symbol
        commas = True to use commas, otherwise use periods'''
        if s in self.depVars():
            if self.type=='xs':
                varlist = {'delta_aspect':'$\Delta$ XS height/width'
                           , 'delta_xshift':'$\Delta$ XS right heaviness'
                           , 'delta_yshift':'$\Delta$ XS bottom heaviness'
                           , 'delta_h_n':'$\Delta$ XS height/original height'
                           , 'delta_w_n':'$\Delta$ XS width/original width'
                           , 'delta_xc_n':'XS right shift/$d_{est}$'
                           }
            elif self.type=='vert' or self.type=='horiz':
                return self.depVarSpl(s)
        elif s.endswith('Ratio') or s.endswith('Prod'):
            if s.endswith('Ratio'):
                symb = '/'
                var1 = s[:-5]
            else:
                symb = r'\times '
                var1 = s[:-4]
            inksymb = self.indVarSymbol(var1, 'ink', commas=commas)[:-1]
            supsymb = self.indVarSymbol(var1, 'sup', commas=commas)[1:]
            return f'{inksymb}{symb}{supsymb}'
        elif s=='int_Ca':
            return r'$Ca=v_{ink}\eta_{sup}/\sigma$'
        elif s.startswith('ink_') or s.startswith('sup_'):
            fluid = s[:3]
            var = s[4:]
            return self.indVarSymbol(var, fluid, commas=commas)
        else:
            if s=='pressureCh0':
                return 'Extrusion pressure (Pa)'
            else:
                return s

        if not s in varlist:
            return s
        if lineType:
            return varlist[s]
        else:
            s1 = varlist[s]
            typ = re.split('_', s)[0]
            s1 = s1[len(typ)+1:]
            return s1