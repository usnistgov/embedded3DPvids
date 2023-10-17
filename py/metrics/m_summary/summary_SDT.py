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
        elif 'under' in self.file.lower():
            self.type = 'under'
        else:
            raise ValueError('Cannot identify print type')
        self.importStillsSummary(diag=diag)
        
    def metaVars(self) -> list:
        '''the variables that are pure metadata, not variables'''
        out = ['bn', 'calibFile', 'date', 'fluFile', 'printFolder']
        for fluid in ['ink', 'sup']:
            out = out + [f'{fluid}_{var}' for var in ['base', 'days', 'dye', 'rheModifier', 'shortname', 'surfactant', 'surfactantWt', 'type', 'var']]
        return out
    
    def displayShort(self, df:pd.DataFrame, yvar:str, xvar:str='int_Ca') -> pd.DataFrame:
        '''display the dataframe with just a few columns'''
        row2 = df[['ink_shortname','sup_shortname', xvar, 'spacing', yvar]].copy()
        display(row2.sort_values(by=yvar))
        return df
    
    def reduceRows(self, yvar:str, *varargs, yvarmin:float=-10**20, yvarmax:float=10*20, **kwargs) -> pd.DataFrame:
        '''each vararg is a boolean series of self.ss, get just the rows that fit all subsets, display, and return the rows'''
        rows = self.ss.copy()
        if len(varargs)>0:
            rows = rows[pd.DataFrame(varargs).all()]
        rows = rows[(rows[yvar]>yvarmin)&(rows[yvar]<yvarmax)]
        return self.displayShort(rows, yvar, **kwargs)

        
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
        
    def printDeps(self, deps:list) -> None:
        '''sort the dependent variables by type and print them out'''
        pos = ['x0', 'xf', 'xc', 'y0', 'yf', 'yc', 'yBot', 'xLeft', 'yTop', 'xRight']
        fusion = ['segments', 'roughness', 'emptiness']
        dims = ['w', 'wn', 'h', 'hn', 'ldiff', 'meanT', 'aspect', 'aspectI', 'xshift', 'yshift', 'area', 'stdevT', 'minmaxT']
        gaps = ['dxprint', 'dx0', 'dxf', 'space_a', 'space_at', 'dy0l', 'dyfl', 'dy0lr', 'dyflr', 'space_l', 'space_b']
        print('\033[1mDependents:\033[0m ')
        d = set(deps)
        for key,l in {'Position':pos, 'Dimensions':dims, 'Fusion':fusion, 'Gaps':gaps}.items():
            li = d.intersection(set(l))
            if len(li)>0:
                print(f'\t\033[31m', '{:<12}:'.format(key), '\033[0m\t',  ', '.join(list(li)))
            d = d.difference(li)
        if len(d)>0:
            print(f'\t\033[31m', '{:<12}:'.format('Unsorted'), '\033[0m\t',  ', '.join(list(d)))
    
    def printKeyTable(self): 
        '''print the dependent variables gridded by title'''
        c = []
        for var in self.depVars():
            s = self.strip(var)
            if not s in c:
                c.append(s)
        self.printDeps(c)
        d1 = {'wp':dict([[i, f'X_w{i}p'] for i in range(1,4)])
                , 'wo':dict([[i, f'X_w{i}o'] for i in range(1,4)])}
        if self.type!='xs':
            d1['dw/dt']=dict([[i, f'dXdt_w{i}o'] for i in range(1,4)])
        d1 = {**d1, **{'wrelax':dict([[i, f'delta_X_w{i}relax'] for i in range(1,4)])
                    , 'write':dict([[i, f'delta_X_write{i}'] for i in range(1,3)])
                    , 'dp':dict([[i, f'X_d{i}p'] for i in range(1,3)])
                    , 'do':dict([[i, f'X_d{i}o'] for i in range(1,3)])}}
        if self.type!='xs':
            d1['dd/dt']=dict([[i, f'dXdt_d{i}o'] for i in range(1,3)])
        d1 = {**d1,**{'drelax':dict([[i, f'delta_X_d{i}relax'] for i in range(1,3)])
                    , 'disturb':dict([[i, f'delta_X_disturb{i}'] for i in range(1,3)])}}          
        
        self.keyTable = pd.DataFrame(d1)
        self.keyTable.fillna('', inplace=True)
        self.keyTable = self.keyTable.T
        display(self.keyTable)
        
    def keyTableVar(self, yvar:str) -> None:
        kt = self.keyTable.copy()
        cols = self.keyTable.columns
        for i,row in self.keyTable.iterrows():
            for col in cols:
                val = row[col]
                if len(val)>0:
                    v2 = val.replace('X', yvar)
                    if v2 in self.ss:
                        kt.loc[i,col] = v2
                    else:
                        kt.loc[i,col] = ''
        display(kt)

    def addRatios(self, **kwargs) -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        return super().addRatios(self.firstDepCol(), **kwargs)

    def addLogs(self, varlist:List[str], **kwargs) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        return super().addLogs(self.firstDepCol(), varlist, **kwargs)
    
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
            varlist = {'x0':'$x_{left}$', 'xf':'$x_{right}$', 'xc':'$x_{center}$'
                       , 'dxprint':'x shift under nozzle'
                       , 'segments':'segments', 'w':'width', 'h':'length', 'hn':'length/intended'
                       , 'roughness':'roughness', 'emptiness':'emptiness'
                       , 'meanT':'ave thickness', 'stdevT':'stdev thickness'
                       , 'minmaxT':'thickness variation', 'ldiff':'shrinkage diference'
                       , 'dx0dt':'left edge shift/time', 'dxfdt':'$\Delta x_{right}$/time', 'dxcdt':'$\Delta x_{center}$/time'
                       , 'dwdt':'widening/time', 'dhdt':'lengthening/time'
                       , 'dsegmentsdt':'rupturing/time', 'dldiffdt':'evening/time'
                       , 'droughnessdt':'roughening/time', 'demptinessdt':'emptying/time'
                       , 'dmeanTdt':'thickening/time', 'dstdevTdt':'d(stdev thickness)/dt'
                       , 'dminmaxTdt':'d(thickness variation)/time'}            
        elif self.type=='xs':
            varlist = {'yBot':'$y_{bottom}$', 'xLeft':'$x_{left}$'
                       , 'segments':'segments', 'aspect':'h/w'
                       , 'xshift':'right-heaviness', 'yshift':'bottom-heaviness'
                       , 'area':'area', 'yTop':'$y_{top}$', 'xRight':'$x_{right}$'
                       , 'xc':'$x_{center}$', 'yc':'$y_{center}$'
                       , 'w':'width', 'h':'height'
                       , 'emptiness':'emptiness', 'roughness':'roughness'
                       , 'aspectI':'h/w/intended'}
        elif self.type=='horiz':
            varlist = {'yBot':'$y_{bottom}$', 'yTop':'$y_{top}$', 'yc':'$y_{center}$'
                       , 'segments':'segments', 'w':'length', 'wn':'length/intended', 'h':'height'
                       , 'roughness':'roughness', 'emptiness':'emptiness'
                       , 'meanT':'ave thickness', 'stdevT':'stdev thickness', 'minmaxT':'thickness variation'
                       , 'ldiff':'shrinkage difference'
                       , 'dy0l':'$y_{top}$ shift behind/under',  'dyfl':'$y_{bottom}$ shift behind/under'
                       , 'dy0lr':'$y_{top}$ shift behind/ahead', 'dyflr':'$y_{bot}$ shift behind/ahead'
                       , 'space_l':'vert space behind/nozzle', 'space_b':'space under/nozzle'
                       , 'dsegmentsdt':'rupturing/time'
                       , 'dyBotdt':'$\Delta y_{bot}$/time', 'dyTopdt':'$\Delta y_{top}$/time'
                       , 'dwdt':'lengthening/time', 'dhdt':'widening/time'
                       , 'dycdt':'y shift/time', 'droughnessdt':'roughening/time'
                       , 'demptiness/dt':'emptying/time', 'dmeanTdt':'thickening/time'
                       , 'dstdevTdt':'d(stdev thickness)/dt', 'dminmaxTdt':'d(thickness variation)/time'}
        elif self.type=='under':
            varlist = {'y0':'$y_{near}$', 'yf':'$y_{far}$', 'yc':'$y_{center}$'
                       , 'segments':'segments', 'w':'length', 'wn':'length/intended', 'h':'width'
                       , 'roughness':'roughness', 'emptiness':'emptiness'
                       , 'meanT':'ave thickness', 'stdevT':'stdev thickness', 'minmaxT':'thickness variation'
                       , 'ldiff':'shrinkage difference'
                       , 'dy0l':'$y_{near}$ shift behind/adj',  'dyfl':'$y_{far}$ shift behind/adj'
                       , 'dy0lr':'$y_{near}$ shift behind/ahead', 'dyflr':'$y_{far}$ shift behind/ahead'
                       , 'space_l':'horiz space behind nozzle', 'space_b':'space next to nozzle'
                       , 'dsegmentsdt':'rupturing/time'
                       , 'dy0dt':'$\Delta y_{near}$/time', 'dy0dt':'$\Delta y_{near}$/time'
                       , 'dwdt':'lengthening/time', 'dhdt':'widening/time'
                       , 'dycdt':'lateral shift/time', 'droughnessdt':'roughening/time'
                       , 'demptiness/dt':'emptying/time', 'dmeanTdt':'thickening/time'
                       , 'dstdevTdt':'d(stdev thickness)/dt', 'dminmaxTdt':'d(thickness variation)/time'}
        else:
            varlist = {}
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
                return self.depVarSpl(s)
            elif self.type=='vert' or self.type=='horiz' or self.type=='under':
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
            if inksymb=='$'+var1+'_{ink}':
                return self.indVarSymbol(s, '', commas=commas)
            else:
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
        
    def depCorrelations(self, fn:str=''):
        '''get a table of spearman correlation strengths between all dependent variables'''
        v = self.depVars()
        out = []
        for i,var1 in enumerate(v):
            for var2 in v[i+1:]:
                s1 = self.strip(var1)
                s2 = self.strip(var2)
                if not s1==s2 and not f'{s1}n'==s2 and not f'{s2}n'==s1:      
                    spear = self.depCorrelation0(var1, var2)
                    out.append(spear)
        self.depCor = pd.DataFrame(out)
        self.exportDepCorrs(fn)
        
    def exportDepCorrs(self, fn:str=''):
        '''export the dependent correlations to file'''
        if os.path.exists(os.path.dirname(fn)):
            plainExp(fn, self.depCor, {}, index=None)
        
    def depCorrelation0(self, var1:str, var2:str) -> dict:
        '''get a single correlation between two variables'''
        spear = rg.spearman(self.ss, var1, var2)
        spear['var1'] = var1
        spear['var2'] = var2
        return spear
        
    def depCorrelation(self, var1:str, var2:str, plot:bool=False):
        '''get a single correlation between two variables'''
        if plot:
            self.ss.plot.scatter(var1, var2)
        if hasattr(self, 'depCor'):
            row = self.depCor[(self.depCor.var1==var1)&(self.depCor.var2==var2)]
            if len(row)==0:
                row = self.depCor[(self.depCor.var1==var2)&(self.depCor.var2==var1)]
            if len(row)>0:
                return dict(row.iloc[0])
        return self.depCorrelation0(var1, var2)