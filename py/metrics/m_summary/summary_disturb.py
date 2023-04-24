#!/usr/bin/env python
'''Functions for collecting data from stills of single disturbed lines, for a whole folder'''

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

class summaryDisturb(summaryMetric):
    '''holds data and functions for handling metric summary tables for disturbed lines'''
    
    def __init__(self, file:str, diag:bool=False):
        super().__init__(file)
        self.file = file
        if 'xs' in self.file:
            self.type = 'xs'
        elif 'vert' in self.file:
            self.type = 'vert'
        elif 'horiz' in self.file:
            self.type = 'horiz'
        self.importStillsSummary(diag=diag)
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        '''import the stills summary and convert sweep types, capillary numbers'''
        self.ss,self.u = plainIm(self.file, ic=False)
        if diag:
            self.printStillsKeys(self.ss)
        return self.ss,self.u

    def firstDepCol(self) -> str:
        '''get the name of the first dependent column'''
        if self.type=='xs':
            return 'delta_aspect'
        elif self.type=='vert':
            return 'bot_delta_segments'
        elif self.type=='horiz':
            return 'delta_segments'
    
    
    def addRatios(self, ss:pd.DataFrame, **kwargs) -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        return super().addRatios(ss, self.firstDepCol(), **kwargs)

    def addLogs(self, ss:pd.DataFrame, varlist:List[str], **kwargs) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        return super().addLogs(ss, self.firstDepCol(), varlist, **kwargs)
    
    def varSymbol(self, s:str, lineType:bool=True, commas:bool=True, **kwargs) -> str:
        '''get a symbolic representation of the variable
        lineType=True to include the name of the line type in the symbol
        commas = True to use commas, otherwise use periods'''
        if self.type=='xs':
            varlist = {'delta_aspect':'$\Delta$ XS height/width'
                       , 'delta_xshift':'$\Delta$ XS right heaviness'
                       , 'delta_yshift':'$\Delta$ XS bottom heaviness'
                       , 'delta_h_n':'$\Delta$ XS height/original height'
                       , 'delta_w_n':'$\Delta$ XS width/original width'
                       , 'delta_xc_n':'XS right shift/$d_{est}$'
                       }
        elif self.type=='vert':
            varlist = {}
            for ltype in ['bot', 'top']:
                varlist = {**varlist, 
                           **{f'{ltype}_delta_segments':f'{ltype} $\Delta$ vert segments'
                        , f'{ltype}_delta_roughness':f'{ltype} $\Delta$ vert roughness'
                        , f'{ltype}_delta_h_n':f'{ltype} $\Delta$ vert length/original length'
                        , f'{ltype}_delta_meanT_n':f'{ltype} $\Delta$ vert thickness/original thickness'
                        , f'{ltype}_delta_xc_n':f'{ltype}'+' vert right shift/$d_{est}$'
                        , f'{ltype}_w_dxprint':f'{ltype}'+' vert writing right shift under nozzle/$d_{est}$'
                        , f'{ltype}_d_dxprint':f'{ltype}'+' vert disturb right shift under nozzle/$d_{est}$'
                        , f'{ltype}_d_dxf':f'{ltype}'+' vert disturb left shift/$d_{est}$'
                        , f'{ltype}_d_space_at':f'{ltype}'+' vert disturb gap at nozzle tip/$d_{est}$'
                        , f'{ltype}_d_space_a':f'{ltype}'+' vert disturb gap next to nozzle/$d_{est}$'
                       }}
        elif self.type=='horiz':
            varlist = {'delta_segments':'$\Delta$ horiz segments'
                    , 'delta_roughness':'$\Delta$ horiz roughness'
                    , 'delta_totlen_n':'$\Delta$ horiz length/original length'
                    , 'delta_meanT_n':'$\Delta$ horiz thickness/original thickness'
                    , 'delta_yc_n':'horiz down shift/$d_{est}$'
                    , 'w_dy0l':'horiz writing up shift behind nozzle/$d_{est}$'
                    , 'd_dy0l':'horiz disturb up shift behind nozzle/$d_{est}$'
                    , 'd_dy0r':'horiz disturb down shift under nozzle/$d_{est}$'
                    , 'd_dy0lr':'horiz disturb net up shift/$d_{est}$'
                    , 'd_space_b':'horiz disturb space under nozzle/$d_{est}$'
                   }
        elif s.endswith('Ratio') or s.endswith('Prod'):
            if s.endswith('Ratio'):
                symb = '/'
                var1 = s[:-5]
            else:
                symb = r'\times '
                var1 = s[:-4]
            return indVarSymbol(var1, 'ink', commas=commas)[:-1]+symb+indVarSymbol(var1, 'sup', commas=commas)[1:]
        elif s=='int_Ca':
            return r'$Ca=v_{ink}\eta_{sup}/\sigma$'
        elif s.startswith('ink_') or s.startswith('sup_'):
            fluid = s[:3]
            var = s[4:]
            return indVarSymbol(var, fluid, commas=commas)
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
