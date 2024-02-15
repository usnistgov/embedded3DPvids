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
import tools.regression as rg
from summary_ideals import *
import file.file_handling as fh
from config import cfg

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
        fh.openExplorer(self.folderName(i))
        
    def folderName(self, i:int):
        return os.path.join(cfg.path.server, self.ss.loc[i,'printFolderR'])
        
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
            
    def indVarSymbol(self, var:str, fluid:str, commas:bool=False) -> str:
        '''get the symbol for an independent variable, eg. dnorm, and its fluid, e.g ink
        commas = True to use commas, otherwise use periods'''
        if len(fluid)>1:
            fluid = fluid[0]
        if commas:
            com = ','
        else:
            com = ''
        if var[-3:]=='adj':
            last = var[-5]
        else:
            last = var[-1]
        if not last in ['a', 'd']:
            last = ''
        if var=='visc' or var=='visc0':
            if len(fluid)>0:
                return '$\eta_{'+fluid+'}$'
            else:
                return '$\eta$'
        elif var=='tau0':
            return '$\tau_{y'+com+fluid+com+last+'}$'
        elif var[:3]=='dPR':
            return '$d_{PR'+com+fluid+com+last+'}$'
        elif var[:5]=='dnorm':
            if var[-3:]=='Inv':
                return '$1/\overline{d_{PR'+com+fluid+com+last+'}}$'
            elif var[-3:]=='adj':
                last = f'{last}a'
            else:
                last = f'{last}e'
            return '$DR_{'+fluid+com+last+'}$'
        elif var[:2]=='Bm':
            return '$Bm_{'+fluid+com+last+'}$'
        elif var=='rate':
            if len(fluid)>0:
                return '$\dot{\gamma}_{'+fluid+'}$'
            else:
                return '$\dot{\gamma}$'
        elif var=='val':
            return {'ink':'ink', 'sup':'support'}[fluid]
        elif var[:5]=='Gstor':
            if len(fluid)>0:
                return '$G\'_{'+fluid+com+last+'}$'
            else:
                return '$G\'$'
        elif var[:4]=='tau0':
            return r'$\tau_{y'+com+fluid+com+last+'}$'
        elif var=='GaRatio':
            return '$G\'_{i'+com+'a}/G\'_{s'+com+'a}$'
        elif var=='GdRatio':
            return '$G\'_{i'+com+'d}/G\'_{s'+com+'d}$'
        elif var=='GtaRatio':
            return '$G\'_{i'+com+r'a}/\tau_{y'+com+'s'+com+'a}$'
        elif var=='tau0aRatio':
            return r'$\tau_{y'+com+'i'+com+r'a}/\tau_{y'+com+'s'+com+'a}$'
        elif var=='tau0dRatio':
            return r'$\tau_{y'+com+'i'+com+r'd}/\tau_{y'+com+'s'+com+'d}$'
        elif var=='tGdRatio':
            return r'$\tau_{i'+com+'d}/G\'_{s'+com+'d}$'
        elif var=='spacing_adj':
            return 'adjusted spacing'
        
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
    
    def printList(self, f, iv:list, title:str) -> None:
        '''filter the list iv, sort, and print'''
        l = list(filter(f, iv))
        l = sorted(l, key=str.casefold)
        l1 = list(filter(lambda x:(self.u[x]==''), l))
        l2 = list(filter(lambda x:(not self.u[x]==''), l))
        if len(l1)>0:
            print('\t\033[31m','{:<12}:'.format(title), '\033[0m',  ', '.join(l1))
        if len(l2)>0:
            print('\t',"{:<12}".format(' '), ' ',  ', '.join(l2))
    
    def printIndeps(self) -> None:
        iv = self.indepVars()
        mv = self.metaVars()
        const = list(filter(lambda x: not x in mv, iv))
        for s in [f'{fluid}_dnorm{dire}_adj' for fluid in ['ink', 'sup'] for dire in ['a', 'd']]:
            if s in self.ss:
                const.append(s)
        print('\033[1mIndependents:\033[0m ')  
        for key,l in {'meta':mv, 'const':const}.items():
            self.printList(lambda x:(not 'sup' in x and not 'ink' in x), l, f'{key}    ')
            self.printList(lambda x:('sup' in x), l, f'{key} sup')
            self.printList(lambda x:('ink' in x), l, f'{key} ink')
        
        
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
    
    def depVars(self) -> List[str]:
        k = self.ss.keys()
        firstCol = self.firstDepCol()
        deps = k[firstCol:]
        deps = deps[~(deps.str.endswith('_SE'))]
        deps = deps[~(deps.str.endswith('_N'))]
        adjustments = [f'{fluid}_dnorm{dire}_adj' for fluid in ['ink', 'sup'] for dire in ['a', 'd']]
        deps = list(set(deps).difference(set(adjustments)))
        return deps
    
    def numericDepVars(self) -> List[str]:
        '''get dependent variables with numeric values'''
        out = []
        deps = self.depVars()
        for dep in deps:
            l = self.ss[dep].dropna().unique()
            if len(l)>1 and self.ss[dep].dtype in ['float64', 'int']:
                out.append(dep)
        return out
    
    def depCorrelations(self):
        '''get a table of spearman correlation strengths between all dependent variables'''
        v = self.numericDepVars()
        out = []
        for i,var1 in enumerate(v):
            for var2 in v[i+1:]:
                spear = rg.spearman(self.ss, var1, var2)
                spear['var1'] = var1
                spear['var2'] = var2
                out.append(spear)
        self.depCor = pd.DataFrame(out)
        