#!/usr/bin/env python
'''Functions for summarizing data from all folders'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from tools.plainIm import *
from summary_ideals import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)

#----------------------------------------------
    
class summaryMetric:
    '''holds data and functions for handling metric summary tables'''
    
    def __init__(self, file:str):
        self.file = file
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        self.ss, self.u = plainIm(self.file)
        
        
    def flipInv(self, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh']) -> None:
        '''find inverse values and invert them (e.g. WeInv)'''
        k = self.ss.keys()
        idx = self.idx0(k)
        for j, s2 in enumerate(varlist):
            for i,s1 in enumerate(['sup', 'ink']):
                xvar = f'{s1}_{s2}'
                if f'{s1}_{s2}Inv' in self.ss and not xvar in self.ss:
                    self.ss.insert(idx, xvar, 1/self.ss[f'{s1}_{s2}Inv'])
                    idx+=1
        if 'int_Ca' not in self.ss:
            self.ss.insert(idx, 'int_Ca', 1/self.ss['int_CaInv'])
            
    def indVarSymbol(self, var:str, fluid:str, commas:bool=True) -> str:
        '''get the symbol for an independent variable, eg. dnorm, and its fluid, e.g ink
        commas = True to use commas, otherwise use periods'''
        if commas:
            com = ','
        else:
            com = '.'
        if var=='visc' or var=='visc0':
            return '$\eta_{'+fluid+'}$'
        elif var=='tau0':
            return '$\tau_{y'+com+fluid+'}$'
        elif var=='dPR':
            return '$d_{PR'+com+fluid+'}$'
        elif var=='dnorm':
            return '$\overline{d_{PR'+com+fluid+'}}$'
        elif var=='dnormInv':
            return '$1/\overline{d_{PR'+com+fluid+'}}$'
        elif var=='rate':
            return '$\dot{\gamma}_{'+fluid+'}$'
        elif var=='val':
            return {'ink':'ink', 'sup':'support'}[fluid]
        else:
            if var.endswith('Inv'):
                varsymbol = '1/'+var[:-3]
            else:
                varsymbol = var
            return '$'+varsymbol+'_{'+fluid+'}$'
        
    def addRatios(self, ss:pd.DataFrame, startName:str, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, s2 in enumerate(varlist):
            xvar =  f'{s2}{operator}'
            if not xvar in ss:
                if not f'ink_{s2}' in ss or not  'sup_{s2}' in ss:
                    self.flipInv()
                if operator=='Prod':
                    ss.insert(idx, xvar, ss[f'ink_{s2}']*ss[f'sup_{s2}'])
                elif operator=='Ratio':
                    ss.insert(idx, xvar, ss[f'ink_{s2}']/ss[f'sup_{s2}'])
                idx+=1
        return ss

    def addLogs(self, ss:pd.DataFrame, startName:str, varlist:List[str]) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, s2 in enumerate(varlist):
            xvar = f'{s2}_log'
            if not xvar in s2:
                ss.insert(idx, xvar, np.log10(ss[s2]))
                idx+=1
        return ss
    
    def printStillsKeys(self) -> None:
        '''sort the keys into dependent and independent variables and print them out'''
        k = self.ss.keys()
        firstCol = self.firstDepCol()
        controls = k[:firstCol]
        deps = k[firstCol:]
        deps = deps[~(deps.str.endswith('_SE'))]
        deps = deps[~(deps.str.endswith('_N'))]
        print(f'Independents: {list(controls)}')
        print()
        print(f'Dependents: {list(deps)}')
        
    def indepVars(self) -> list:
        k = self.ss.keys()
        firstCol = self.firstDepCol()
        controls = k[:firstCol]
        return list(controls)
    
    def printIndeps(self) -> None:
        iv = self.indepVars()
        print('\033[1mIndependents:\033[0m ')
        print('\t', ', '.join(filter(lambda x:(not 'sup' in x and not 'ink' in x), iv)))
        print('\t',  ', '.join(filter(lambda x:('sup' in x), iv)))
        print('\t',  ', '.join(filter(lambda x:('ink' in x), iv)))
        
        
    def idx(self, k:list, name:str) -> int:
        '''find first index that matches name'''
        if name in k:
            return int(np.argmax(k==name))
        else:
            return 1
        
    def firstDepCol(self) -> int:
        '''index of the first dependent variable column'''
        k = self.ss.keys()
        return np.argmax(k.str.endswith('_SE'))-1
        
    def idx0(self, k:list) -> int:
        '''get the index of the first dependent variable'''
        return self.idx(k, self.firstDepCol())
    
    def depVars(self):
        k = self.ss.keys()
        firstCol = self.firstDepCol()
        deps = k[firstCol:]
        deps = deps[~(deps.str.endswith('_SE'))]
        deps = deps[~(deps.str.endswith('_N'))]
        return deps
    