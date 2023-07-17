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
import file.file_handling as fh

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
        if not os.path.exists(file):
            raise FileNotFoundError(f'{file} does not exist')
            
    def openFolder(self, i:int):
        '''open the folder given by the row in the ss dataframe'''
        fh.openExplorer(self.ss.loc[i,'printFolder'])
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        self.ss, self.u = plainIm(self.file)
        
    def flipInv(self, ss:pd.DataFrame=[], varlist:list = ['Ca', 'dPR', 'dnorm', 'We', 'Oh']) -> pd.DataFrame:
        '''find inverse values and invert them (e.g. WeInv)'''
        if len(ss)==0:
            ss = self.ss
        k = ss.keys()
        idx = self.idx0(k)
        for j, var in enumerate(varlist):
            for i,fluid in enumerate(['sup', 'ink']):
                for dire in ['', 'a', 'd']:
                    xvar = f'{fluid}_{var}{dire}'
                    if f'{fluid}_{var}Inv{dire}' in ss and not xvar in ss:
                        ss.insert(idx, xvar, 1/ss[f'{fluid}_{var}Inv{dire}'])
                        u = self.u[f'{fluid}_{var}Inv{dire}']
                        if u=='':
                            self.u[xvar] = ''
                        else:
                            self.u[xvar] = f'1/{u}'
                        idx+=1
        if 'int_Ca' not in ss:
            ss.insert(idx, 'int_Ca', 1/ss['int_CaInv'])
            self.u['int_Ca'] = ''
        return ss
            
    def indVarSymbol(self, var:str, fluid:str, commas:bool=True) -> str:
        '''get the symbol for an independent variable, eg. dnorm, and its fluid, e.g ink
        commas = True to use commas, otherwise use periods'''
        if commas:
            com = ','
        else:
            com = '.'
        last = var[-1]
        if not last in ['a', 'd']:
            last = ''
        else:
            last = com+{'a':'asc', 'd':'desc'}[last]
        if var=='visc' or var=='visc0':
            return '$\eta_{'+fluid+'}$'
        elif var=='tau0':
            return '$\tau_{y'+com+fluid+'}$'
        elif var[:3]=='dPR':
            return '$d_{PR'+com+fluid+last+'}$'
        elif var[:5]=='dnorm':
            return '$\overline{d_{PR'+com+fluid+last+'}}$'
        elif var[:5]=='dnorm' and var[-3]=='Inv':
            return '$1/\overline{d_{PR'+com+fluid+last+'}}$'
        elif var[:2]=='Bm':
            return '$Bm_{'+fluid+last+'}$'
        elif var=='rate':
            return '$\dot{\gamma}_{'+fluid+'}$'
        elif var=='val':
            return {'ink':'ink', 'sup':'support'}[fluid]
        elif var[:5]=='Gstor':
            return '$G\'_{'+fluid+last+'}$'
        elif var[:4]=='tau0':
            return r'$\tau_{y'+com+fluid+last+'}$'
        elif var=='GaRatio':
            return '$G\'_{ink'+com+'a}/G\'_{sup'+com+'a}$'
        elif var=='GdRatio':
            return '$G\'_{ink'+com+'d}/G\'_{sup'+com+'d}$'
        elif var=='GtaRatio':
            return '$G\'_{ink'+com+r'a}/\tau_{y'+com+'sup'+com+'a}$'
        elif var=='tau0aRatio':
            return r'$\tau_{y'+com+'ink'+com+r'a}/\tau_{y'+com+'sup'+com+'a}$'
        elif var=='tau0dRatio':
            return r'$\tau_{y'+com+'ink'+com+r'd}/\tau_{y'+com+'sup'+com+'d}$'
        elif var=='tGdRatio':
            return r'$\tau_{ink'+com+'d}/G\'_{sup'+com+'d}$'
        else:
            if var.endswith('Inv'):
                varsymbol = '1/'+var[:-3]
            else:
                varsymbol = var
            if len(fluid)>0:
                return '$'+varsymbol+'_{'+fluid+'}$'
            else:
                return varsymbol
        
    def addRatios(self, startName:str='', ss:pd.DataFrame=[], varlist:list = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        if len(ss)==0:
            ss = self.ss
        if startName=='':
            startName = self.firstDepCol()
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, var in enumerate(varlist):
            for dire in ['', 'a', 'd']:   # ascending, descending
                if f'ink_{var}{dire}' in ss:
                    xvar =  f'{var}{dire}{operator}'
                    if not xvar in self.ss:
                        if not f'ink_{var}{dire}' in ss or not f'sup_{var}{dire}' in ss:
                            ss = self.flipInv(ss, varlist=[f'{var}{dire}'])
                        if operator=='Prod':
                            ss.insert(idx, xvar, ss[f'ink_{var}{dire}']*ss[f'sup_{var}{dire}'])
                        elif operator=='Ratio':
                            ss.insert(idx, xvar, ss[f'ink_{var}{dire}']/ss[f'sup_{var}{dire}'])
                        idx+=1
        return ss

    def addLogs(self, startName:str='', varlist:List[str]=[], ss:pd.DataFrame=[]) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe. this is useful for adding values to subsets of dataframes as well'''
        if len(ss)==0:
            ss = self.ss
        k = ss.keys()
        if startName=='':
            startName = self.firstDepCol()
        idx = int(np.argmax(k==startName))
        for j, var in enumerate(varlist):
            xvar = f'{var}_log'
            if not xvar in var:
                ss.insert(idx, xvar, np.log10(ss[var]))
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
    
    def printList(self, f, iv:list) -> None:
        '''filter the list iv, sort, and print'''
        l = list(filter(f, iv))
        l = sorted(l, key=str.casefold)
        l1 = list(filter(lambda x:(self.u[x]==''), l))
        l2 = list(filter(lambda x:(not self.u[x]==''), l))
        if len(l1)>0:
            print('\t',  ', '.join(l1))
        if len(l2)>0:
            print('\t',  ', '.join(l2))
    
    def printIndeps(self) -> None:
        iv = self.indepVars()
        mv = self.metaVars()
        const = list(filter(lambda x: not x in mv, iv))
        print('\033[1mIndependents:\033[0m ')  
        for l in [mv, const]:
            self.printList(lambda x:(not 'sup' in x and not 'ink' in x), l)
            self.printList(lambda x:('sup' in x), l)
            self.printList(lambda x:('ink' in x), l)
        
        
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
    