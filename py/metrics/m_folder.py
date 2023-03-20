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
from tools.plainIm import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims
import file_handling as fh
from m_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
    
    
class metricSummary:
    '''holds data and functions for handling metric summary tables'''
    
    def __init__(self, file:str):
        self.file = file
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        self.ss, self.u = plainIm(self.file)
        
        
    def addRatios(self, ss:pd.DataFrame, startName:str, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, s2 in enumerate(varlist):
            xvar =  f'{s2}{operator}'
            if not xvar in ss:
                if not f'ink_{s2}' in ss or not  'sup_{s2}' in ss:
                    ss = flipInv(ss)
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
    
    def printStillsKeys(self, ss:pd.DataFrame) -> None:
        '''sort the keys into dependent and independent variables and print them out'''
        k = ss.keys()
        k = k[~(k.str.endswith('_SE'))]
        k = k[~(k.str.endswith('_N'))]
        idx = self.idx0(k)
        controls = k[:idx]
        deps = k[idx:]
        print(f'Independents: {list(controls)}')
        print()
        print(f'Dependents: {list(deps)}')
        
    def idx(self, k:list, name:str) -> int:
        if name in k:
            return int(np.argmax(k==name))
        else:
            return 1
        
    def idx0(self, k:list) -> int:
        '''get the index of the first dependent variable'''
        return self.idx(k, self.firstDepCol())
    
#----------------------------------------------

class disturbMeasures:
    '''for a folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwriteMeasure:bool=False, overwriteSummary:bool=False, lineType:str='horiz', diag:int=0, **kwargs) -> None:
        self.folder = folder
        self.overwriteMeasure = overwriteMeasure
        self.overwriteSummary = overwriteSummary
        self.pfd = fh.printFileDict(folder)
        self.lineType = lineType
        self.fn = self.pfd.newFileName(f'{lineType}Measure', '.csv')
        self.failfn = self.pfd.newFileName(f'{lineType}Failures', '.csv')
        self.diag = diag
        self.kwargs = kwargs
        self.depVars = []

    def measure(self, lines:List[str], func) -> int:
        '''after initialize file name, measure all lines'''
        if not hasattr(self, 'fn'):
            raise ValueError(f'Missing export file name in {self.folder}')
        if os.path.exists(self.fn) and not self.overwriteMeasure:
            return 1
        
        files = {}
        for f in os.listdir(self.folder):
            if 'vstill' in f:
                files[re.split('_', re.split('vstill_', f)[1])[1]] = os.path.join(self.folder, f)

        self.du = {}
        out = []
        failures = []
        
        for file in files.values():
            m, u = func(file, pfd=self.pfd, diag=self.diag-1, **self.kwargs)
            if len(u)>len(self.du):
                self.du = u
            if len(m)==1:
                failures.append(file)
            out.append(m)
        self.df = pd.DataFrame(out)
        self.failures = pd.DataFrame({'file':failures})
        plainExp(self.failfn, self.failures, {'file':''})
        plainExp(self.fn, self.df, self.du)
        return 0
    
    def importMeasure(self):
        '''import the table of measurements of each image'''
        if hasattr(self, 'df'):
            self.importFailures()
            return
        if not hasattr(self, 'pfd'):
            self.pfd = fh.printFileDict(self.folder)
        measureFile = f'{self.lineType}Measure'
        if hasattr(self.pfd, measureFile) and not self.overwriteMeasure:
            self.df, self.du = plainIm(getattr(self.pfd, measureFile), ic=0)
            self.df.line.fillna('', inplace=True)
        else:
            self.measureFolder()
        self.importFailures()
            
    def importFailures(self):
        '''import the list of failed files'''
        if not hasattr(self, 'failures'):
            if os.path.exists(self.failfn):
                self.failures, _ = plainIm(self.failfn, ic=0)
            else:
                self.failures = pd.DataFrame([])
        
        
    def summaryHeader(self) -> int:
        '''top matter for summaries. return 0 if we already have a summary, return 1 if we have measurements, return 2 if we cannot proceed'''
        retval = 2
        self.summaryFn = self.pfd.newFileName(f'{self.lineType}Summary', '.csv')
        if os.path.exists(self.summaryFn) and not self.overwriteSummary:
            self.summary, self.summaryUnits = plainImDict(self.summaryFn, unitCol=1, valCol=2)
            self.importFailures()
            return 0
        self.importMeasure()
        if hasattr(self, 'df'):
            retval = 1
        else:
            return 2
        self.pv = printVals(self.folder)
        self.pxpmm = self.pv.pxpmm
        self.mr, self.mu = self.pv.metarow()
        return retval
        
        
    def convertValuesAndExport(self, ucombine:dict, lists:dict):
        '''find changes between ovservations'''
        out = {}
        units = {}
        for key,val in lists.items():
            convertValue(key, val, ucombine, self.pxpmm, units, out)
        self.summary = {**self.mr, **out}
        self.summaryUnits = {**self.mu, **units}
        plainExpDict(self.summaryFn, self.summary, units=self.summaryUnits)
        
    def dflines(self, name:str) -> pd.DataFrame:
        return self.df[self.df.line.str.endswith(name)]
        
    def dfline(self, name:str) -> pd.Series:
        '''get the line from the measurements dataframe'''
        wodf = self.dflines(name)
        if not len(wodf)==1:
            return []
        else:
            return wodf.iloc[0]
        
    def addValue(self, results:dict, units:dict, name:str, value:Any, unit:str) -> None:
        '''add the result to the results dict list and units dict'''
        if name in results:
            results[name].append(value)
        else:
            results[name] = [value]
            units[name] = unit
            self.depVars.append(name)
        
    def addValues(self, results:dict, units:dict, name:str, values:list, unit:str) -> None:
        '''add the result to the results dict list and units dict'''
        if name in results:
            results[name] = results[name]+values
        else:
            results[name] = values
            units[name] = unit
            self.depVars.append(name)
        
    def printAll(self):
        '''print all statistics'''
        print(self.folder)
        df = pd.DataFrame()
        for var in self.depVars:
            val = self.summary[var]
            se = self.summary[f'{var}_SE']
            n = self.summary[f'{var}_N']
            units = self.summaryUnits[var]
            df.loc[var, 'value'] = '{:5.4f}'.format(val)
            df.loc[var, 'SE'] = '{:5.4f}'.format(se)
            df.loc[var, 'N'] = n
            df.loc[var, 'units'] = units
        display(df)
        

        
        
class SDTMeasures(disturbMeasures):
    '''for a folder, measure the SDT lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, lineType:str='horiz', **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, lineType=lineType, **kwargs)
        if not f'disturb' in os.path.basename(self.folder):
            return
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
        return self.pg.progDims[self.pg.progDims.name.str.endswith(name)]
    
    def pgline(self, name:str) -> pd.Series:
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

        return out
        

    def singles(self) -> list:
        '''get a list of single values to average across all 4 groups'''
        out = []
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
        out = [f'{ll}{on}' for ll in llist for on in ['', 'o1']]
        return out
        
        
class summaries:
    '''recursively create summaries'''
    
    def __init__(self, topFolder:str, measureFunc, overwrite:bool=False, mustMatch:list=[], **kwargs):
        self.topFolder = topFolder
        self.overwrite = overwrite
        self.measureFunc = measureFunc
        self.out = []
        self.units = {}
        self.errorFiles = []
        self.mustMatch = mustMatch
        self.failures = pd.DataFrame([])
        self.recurse(topFolder, **kwargs)
        
        
    def checkMatch(self, topFolder):
        '''check if the folder has all of the labels required'''
        for s in self.mustMatch:
            if not s in topFolder:
                return False
        return True
        
    def recurse(self, topFolder:str, **kwargs):
        '''measure values, recursing through the folder hierarchy'''
        if not fh.isPrintFolder(topFolder):
            for f in os.listdir(topFolder):
                self.recurse(os.path.join(topFolder, f), **kwargs)
            return
        
        if not self.checkMatch(topFolder):
            # folder doesn't match intended string
            return
        
        try:
            if ('overwriteMeasure' in kwargs and kwargs['overwriteMeasure']) or ('overwriteSummary' in kwargs and kwargs['overwriteSummary']):
                # redo measurements
                summary, units, failures = self.measureFunc(topFolder, overwrite=self.overwrite, **kwargs)
            else:
                # just take the existing measurements
                pfd = fh.printFileDict(topFolder)
                if hasattr(pfd, 'summary'):
                    summary, units = plainImDict(pfd.summary, unitCol=1, valCol=2)
                else:
                    summary = {}
                    units = {}
                if hasattr(pfd, 'failure'):
                    failures, _ = plainIm(pfd.failure, ic=0)
                else:
                    failures = []
        except Exception as e:
            print(f'Error in {topFolder}: {e}')
            traceback.print_exc()
        else:
            if len(summary)>0:
                self.units = {**self.units, **units}
                self.out.append(summary)
        if len(failures)>0:
            self.failures = pd.concat([self.failures, failures])


    def export(self, fn:str) -> None:
        df = pd.DataFrame(self.out)
        plainExp(fn, df, self.units, index=False)
        
    def exportFailures(self, fn:str) -> None:
        '''export a list of failed files'''
        plainExp(fn, self.failures, {}, index=False)
        
#---------------------------------------------------
        
class failureTest:
    '''for testing failed files'''
    
    def __init__(self, failureFile:str, testFunc):
        self.failureFile = failureFile
        self.testFunc = testFunc
        self.importFailures()
        
    def countFailures(self):
        self.bad = self.df[self.df.approved==False]
        self.approved = self.df[self.df.approved==True]
        self.failedLen = len(self.bad)
        self.failedFolders = len(self.bad.fostr.unique())
        print(f'{self.failedLen} failed files, {self.failedFolders} failed folders')
        
    def firstBadFile(self):
        return self.bad.iloc[0].name
    
    def firstBadFolder(self):
        return self.bad.iloc[0]['fostr']
        
    def importFailures(self):
        df, _ = plainIm(self.failureFile, ic=None)
        for i,row in df.iterrows():
            folder = os.path.dirname(row['file'])
            df.loc[i, 'fostr'] = folder.replace(cfg.path.server, '')[1:]
            df.loc[i, 'fistr'] = os.path.basename(row['file'])
            if not 'approved' in row:
                df.loc[i, 'approved'] = False
        self.df = df
        self.countFailures()
        
    def testFile(self, i:int, **kwargs):
        '''test out measurement on a single file, given by the row number'''
        row = self.df.loc[i]
        approved = row['approved']
        print(f'Row {i}, approved {approved}')
        self.testFunc(row['fostr'], row['fistr'], **kwargs)
        
    def testFolder(self, folder:str, **kwargs):
        ffiles = self.bad[self.bad.fostr==folder]
        for i,_ in ffiles.iterrows():
            self.testFile(i, **kwargs)

    def approveFile(self, i:int, export:bool=True):
        '''approve the file'''
        self.df.loc[i, 'approved'] = True
        folder = os.path.join(cfg.path.server, self.df.loc[i, 'fostr'])
        file = self.df.loc[i, 'file']
        
        # change the failure file in this file's folder
        if export:
            pfd = fh.printFileDict(folder)
            if hasattr(pfd, 'failure'):
                df, _ = plainIm(pfd.failure, ic=0)
                df = df[~(df.file==file)]
                plainExp(pfd.failure, df, {})
        self.countFailures()
        
    def approveFolder(self, fostr:str):
        ffiles = self.df[self.df.fostr==fostr]
        for i,_ in ffiles.iterrows():
            self.approveFile(i, export=False)
        pfd = fh.printFileDict(os.path.join(cfg.path.server, fostr))
        if hasattr(pfd, 'failure'):
            plainExp(pfd.failure, pd.DataFrame(['']), {})
        
    def export(self):
        plainExp(self.failureFile, self.df, {}, index=False)
            
      