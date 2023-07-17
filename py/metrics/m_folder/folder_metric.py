'''Functions for collecting data from stills of images, for a whole folder'''

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
from crop_locs import *
from m_tools import *
from tools.plainIm import *
from val.v_print import printVals
from progDim.prog_dim import getProgDims, getProgDimsPV
import file.file_handling as fh
from vid.noz_detect import *
from tools.timeCounter import timeObject
import tools.regression as reg
from folder_size_check import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
            
class folderMetric(timeObject):
    '''for a folder, measure all images
    export a table of values (Measure)
    export a list of failed files (Failures)
    export a row of summary values (Summary)'''
    
    def __init__(self, folder:str, overwriteMeasure:bool=False, overwriteSummary:bool=False, overwriteCropLocs:bool=False, diag:int=0, **kwargs) -> None:
        super().__init__()
        self.folder = folder
        self.overwriteMeasure = overwriteMeasure
        self.overwriteSummary = overwriteSummary
        self.overwriteCropLocs = overwriteCropLocs
        if 'pfd' in kwargs:
            self.pfd = kwargs.pop('pfd')
        else:
            self.pfd = fh.printFileDict(folder)
        if 'pv' in kwargs:
            self.pv = kwargs.pop('pv')
        else:
            self.pv = printVals(self.folder, pfd=self.pfd, **kwargs)
        if 'pg' in kwargs:
            self.pg = kwargs.pop('pg')
        self.diag = diag
        self.kwargs = kwargs
        self.depVars = []
        self.getLineType()
        self.getFNs()

    def getLineType(self) -> None:
        '''get the type of line being printed'''
        bn = os.path.basename(self.folder).lower()
        if 'horiz' in bn:
            self.lineType = 'horiz'
        elif 'vert' in bn:
            self.lineType = 'vert'
        elif 'xs' in bn and '+y' in bn:
            self.lineType = 'xs+y'
        elif 'xs' in bn and '+z' in bn:
            self.lineType = 'xs+z'
        else:
            raise ValueError(f'Failed to identify line type in {self.folder}')
        
    def getFNs(self):
        self.fn = self.pfd.newFileName(f'measure', '.csv')
        self.failfn = self.pfd.newFileName(f'failures', '.csv')
        self.summaryFn = self.pfd.newFileName(f'summary', '.csv')

    def measure(self, fm) -> int:
        '''after initialize file name, measure all lines. fm is a class definition for a fileMetric object'''
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
        self.nd = nozData(self.folder, pfd=self.pfd)
        if not os.path.exists(self.pfd.nozDims):
            self.nd.detectNozzle(export=True)
            self.nd.nozDims()
        if not hasattr(self, 'pg'):
            self.pg  = getProgDimsPV(self.pv)
        self.cl = cropLocs(self.folder, pfd=self.pfd)
        for file in files.values():
            self.nd.resetDims()
            try:
                m, u = fm(file, pfd=self.pfd, pv=self.pv, nd=self.nd, pg=self.pg, cl=self.cl
                          , diag=self.diag-1, exportCropLoc=False, overwriteCropLoc=self.overwriteCropLocs, **self.kwargs).values()
                self.du = {**self.du, **u}
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                failures.append({'file':file, 'error':e})
            else:
                if len(m['line'])<1:
                    if 'error' in m:
                        # this image was whited out. failed intentionally
                        er = m.pop('error')
                        failures.append({'file':file, 'error':er})
                    else:
                        failures.append({'file':file, 'error':'no vals detected'})
                out.append(m)
        self.df = pd.DataFrame(out)
        self.failures = pd.DataFrame(failures)
        plainExp(self.failfn, self.failures, {'file':'', 'error':''})
        plainExp(self.fn, self.df, self.du)
        self.cl.export()
        return 0
    
    def importMeasure(self):
        '''import the table of measurements of each image'''
        if hasattr(self, 'df'):
            self.importFailures()
            return
        if not hasattr(self, 'pfd'):
            self.pfd = fh.printFileDict(self.folder)
        if os.path.exists(self.fn) and not self.overwriteMeasure:
            self.df, self.du = plainIm(self.fn, ic=0)
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
        if os.path.exists(self.summaryFn) and not self.overwriteSummary:
            self.summary, self.summaryUnits = plainImDict(self.summaryFn, unitCol=1, valCol=2)
            self.importFailures()
            return 0
        self.importMeasure()
        if hasattr(self, 'df'):
            retval = 1
        else:
            return 2
        self.pxpmm = self.pv.pxpmm
        self.mr, self.mu = self.pv.metarow()
        self.aves = {}
        self.sems = {}
        self.ns = {}
        self.aveunits = {}  # dictionaries for collecting values to average later
        return retval
    
    def summaryValues(self) -> Tuple[dict,dict, pd.DataFrame]:
        '''get the summary, units, and list of failures'''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = {}
        if hasattr(self, 'summaryUnits'):
            u = self.summaryUnits
        else:
            u = {}
        if hasattr(self, 'failures'):
            f = self.failures
        else:
            f = pd.DataFrame([])
        return s,u,f

    
    def convertValue(self, key:str) -> Tuple:
        '''convert the values from px to mm'''
        uke = self.aveunits[key]
        if uke=='px':
            c = 1/self.pxpmm
            u2 = 'mm'
        elif uke=='px^2':
            c = 1/self.pxpmm**2
            u2 = 'mm^2'
        elif uke=='px^3':
            c = 1/self.pxpmm**3
            u2 = 'mm^3'
        else:
            c = 1
            u2 = uke
        self.summaryUnits[key] = u2
        self.summaryUnits[f'{key}_SE'] = u2
        self.summaryUnits[f'{key}_N'] = ''
        if key in self.sems and key in self.ns:
            m, se, n = pooledSE(self.aves[key], self.sems[key], self.ns[key])
            self.summary[key] = m*c
            self.summary[f'{key}_SE'] = se*c
            self.summary[f'{key}_N'] = n
        else:
            val = np.array(self.aves[key])
            val = val[~np.isnan(val)]
            if len(val)>0:
                self.summary[key] = np.mean(val)*c
                self.summary[f'{key}_SE'] = sem(val)*c
                self.summary[f'{key}_N'] = len(val)
        
    def convertValuesAndExport(self):
        '''find changes between ovservations'''
        self.summary = self.mr.copy()
        self.summaryUnits = self.mu.copy()
        for key in self.aves:
            self.convertValue(key)
        plainExpDict(self.summaryFn, self.summary, units=self.summaryUnits)
        
    def dflines(self, name:str) -> pd.DataFrame:
        if not 'o' in name:
            return self.df[(self.df.line.str.contains(name))&(~self.df.line.str.contains('o'))]
        else:
            return self.df[self.df.line.str.contains(name)]
        
    def dfline(self, name:str) -> pd.Series:
        '''get the line from the measurements dataframe'''
        wodf = self.dflines(name)
        if not len(wodf)==1:
            return []
        else:
            return wodf.iloc[0]
        
    def addValue(self, name:str, value:Any, unit:str) -> None:
        '''add the result to the results dict list and units dict'''
        self.addValues(name, [value], unit)
        
    def addValues(self, name:str, values:list, unit:str) -> None:
        '''add the result to the results dict list and units dict'''
        if name in self.aves:
            self.aves[name] = self.aves[name]+values
        else:
            self.aves[name] = values
            self.aveunits[name] = unit
            self.depVars.append(name)
            
    def addSingle(self, lineName:str, dv:list):
        '''add values for a single line name'''
        # get average values
        lines = self.dflines(lineName)
        for var in dv:
            if var in lines:
                self.addValues(f'{var}_{lineName}', list(lines[var].dropna()), self.du[var])
            
    def addSlopes(self, lineName:str, yvar:str, tunits:str, rcrit:float=0.9) -> None:
        '''for each line group, get the slope over time and add the slope to the results dict list and units dict. only get slopes if regression is good'''
        lines = self.dflines(lineName)
        varname = f'd{yvar}dt_{lineName}'
        if not yvar in self.du:
            return
        uu = self.du[yvar]
        unit = f'{uu}/{tunits}' 
        for gname, df in lines.groupby(['gname']):
            d = reg.regPD(df, ['time'], yvar)
            if len(d)>0 and d['r2']>rcrit:
                slope = d['b']
                if varname in self.aves:
                    self.aves[varname].append(slope)
                else:
                    self.aves[varname] = [slope]
                    self.aveunits[varname] = unit
                    self.depVars.append(varname)
                    

    def addPair(self, dv:list, pair:list) -> None:
        '''for each line group, get the slope over time and add the slope to the results dict list and units dict. only get slopes if regression is good'''
        i1 = pair[0]
        i2 = pair[1]
        nm = pair[2]
        lines1 = self.dflines(i1)
        lines2 = self.dflines(i2)
        for gname, df1 in lines1.groupby(['gname']):
            df2 = lines2[lines2.gname==gname]
            for var in dv:
                # iterate over all dependent variables
                if var in df1:
                    m1, se1, n1 = msen(df1[var])   # mean and standard error of line 1
                    m2, se2, n2 = msen(df2[var])   # mean and standard error of line 2
                    dm = m2-m1
                    dmse = np.sqrt((se1**2+se2**2)/2)
                    n = n1+n2
                    name = f'delta_{var}_{nm}'
                    if name in self.aves:
                        self.aves[name].append(dm)
                        self.sems[name].append(dmse)
                        self.ns[name].append(n)
                    else:
                        self.aves[name] = [dm]
                        self.sems[name] = [dmse]
                        self.ns[name] = [n]
                        self.aveunits[name] = self.du[var]
                        self.depVars.append(name)
                    

    def printAll(self):
        '''print all statistics'''
        print(self.folder)
        df = pd.DataFrame()
        for var in self.depVars:
            if var in self.summary:
                val = self.summary[var]
                se = self.summary[f'{var}_SE']
                n = self.summary[f'{var}_N']
                units = self.summaryUnits[var]
                df.loc[var, 'value'] = '{:5.4f}'.format(val)
                df.loc[var, 'SE'] = '{:5.4f}'.format(se)
                df.loc[var, 'N'] = n
                df.loc[var, 'units'] = units
        display(df)
        
