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
from tools.plainIm import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims
import file.file_handling as fh
from m_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
 
class summarizer(fh.folderLoop):
    '''recursively create measures, summaries, failuares, and collect all of the summaries into a table. measureClass is a class definition for a folderMetric class. failures will be a list of files'''
    
    def __init__(self, folders:Union[str,list], measureClass, overwrite:bool=False, overwriteMeasure:bool=False, overwriteSummary:bool=False, **kwargs):
        super().__init__(folders, self.summarize, **kwargs)
        self.overwrite = overwrite
        self.overwriteMeasure = overwriteMeasure
        self.overwriteSummary = overwriteSummary
        self.measureClass = measureClass
        self.out = []
        self.units = {}
        self.failures = pd.DataFrame([])
        
    def summarize(self, folder:str) -> None:
        '''get summaries from a single folder and add them to the running list'''
        summary = []
        failures = []
        if not (self.overwriteMeasure or self.overwriteSummary):
            pfd = fh.printFileDict(folder)
            if hasattr(pfd, 'summary') and hasattr(pfd, 'failures') and os.path.exists(pfd.summary) and os.path.exists(pfd.failures):
                summary, units = plainImDict(pfd.summary, unitCol=1, valCol=2)
                failures, _ = plainIm(pfd.failures, ic=0)
        
        if len(summary)==0:
            cl = self.measureClass(folder, overwrite=self.overwrite, overwriteMeasure=self.overwriteMeasure, overwriteSummary=self.overwriteSummary, **self.kwargs)
            if self.overwriteMeasure:
                cl.measureFolder()
            if self.overwriteSummary:
                summary, units, failures = cl.summarize()
            else:
                cl.summaryHeader()
                summary, units, failures = cl.summaryValues()

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
        if len(self.folderErrorList)>0:
            plainExp(fn.replace('Failures', 'Errors'), pd.DataFrame(self.folderErrorList), {}, index=False)
            
    def run(self):
        self.out = []
        self.units = {}
        self.failures = pd.DataFrame([])
        super().run()
        
#---------------------------------------------------
        
class failureTest:
    '''for testing failed files for all folders in a topfolder. testFunc should be a fileMetric class definition'''
    
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
    
    def file(self, i):
        row = self.df.loc[i]
        file = os.path.join(cfg.path.server, row['fostr'], row['fistr'])
        return file
    
    def folder(self, i):
        row = self.df.loc[i]
        folder = os.path.join(cfg.path.server, row['fostr'])
        return folder
        
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
        file = os.path.join(cfg.path.server, row['fostr'], row['fistr'])
        self.testFunc(file, **kwargs)
        
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
            if not hasattr(self, 'pfd'):
                self.pfd = fh.printFileDict(folder)
            if hasattr(self.pfd, 'failure'):
                df, _ = plainIm(self.pfd.failure, ic=0)
                df = df[~(df.file==file)]
                plainExp(fn, df, {})
        self.countFailures()
        
    def disableFile(self, i:int) -> None:
        '''overwrite the Usegment and MLsegment files so this file cannot be measured. useful if the function is measuring values from a bad image'''
        file = self.file(i)
        self.testFunc(file, measure=False).disableFile()
        
    def approveFolder(self, fostr:str):
        ffiles = self.df[self.df.fostr==fostr]
        for i,_ in ffiles.iterrows():
            self.approveFile(i, export=False)
        if not hasattr(self, 'pfd'):
            self.pfd = fh.printFileDict(os.path.join(cfg.path.server, fostr))
        if hasattr(self.pfd, 'failure'):
            plainExp(self.pfd.failure, pd.DataFrame(['']), {})
            
    def approveFolderi(self, i:int) -> None:
        fostr = self.df.loc[i, 'fostr']
        self.approveFolder(fostr)
            
    def openFolder(self, i:int) -> None:
        '''open the folder in explorer'''
        folder = self.folder(i)
        fh.openExplorer(folder)
        
    def openPaint(self, i:int) -> None:
        file = self.file(i)
        openInPaint(file)  # from m_tools
        
    def openFolderInPaint(self, fostr:str) -> None:
        ffiles = self.bad[self.bad.fostr==fostr]
        for i,_ in ffiles.iterrows():
            self.openPaint(i)
        
    def export(self):
        plainExp(self.failureFile, self.df, {}, index=False)